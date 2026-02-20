import os
import re
import time
import html
import threading
from typing import Optional, Dict, Tuple, List

import requests
import httpx
from Crypto.Cipher import AES

from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel

# -------------------------
# Config
# -------------------------
BASE_URL = "https://asmodeus.free.nf"
HOME_URL = f"{BASE_URL}/"
WARMUP_URL = f"{BASE_URL}/index.php?i=1"
CHAT_URL = f"{BASE_URL}/deepseek.php"
COOKIE_DOMAIN = "asmodeus.free.nf"

API_KEY: Optional[str] = os.getenv("API_KEY", "20262025")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-V3")

SESSION_TTL_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 60

DEBUG = os.getenv("DEBUG", "1") == "1"

# نماذج معروفة (لأوامر /models و /model)
AVAILABLE_MODELS: List[str] = [
    "DeepSeek-V1",
    "DeepSeek-V2",
    "DeepSeek-V2.5",
    "DeepSeek-V3",
    "DeepSeek-V3-0324",
    "DeepSeek-V3.1",
    "DeepSeek-V3.2",
    "DeepSeek-R1",
    "DeepSeek-R1-0528",
    "DeepSeek-R1-Distill",
    "DeepSeek-Prover-V1",
    "DeepSeek-Prover-V1.5",
    "DeepSeek-Prover-V2",
    "DeepSeek-VL",
    "DeepSeek-Coder",
    "DeepSeek-Coder-V2",
    "DeepSeek-Coder-6.7B-base",
    "DeepSeek-Coder-6.7B-instruct",
]

app = FastAPI(title="DeepSeek Proxy API")

_lock = threading.Lock()
_session: Optional[requests.Session] = None
_session_created_at: float = 0.0

# حفظ نموذج لكل Chat في تيليجرام
_chat_models: Dict[int, str] = {}


# -------------------------
# Schemas
# -------------------------
class ChatReq(BaseModel):
    question: str
    model: Optional[str] = None


# -------------------------
# Helpers
# -------------------------
def log(msg: str):
    if DEBUG:
        print(msg, flush=True)


def clean_html_response(text: str) -> str:
    """
    يحوّل HTML/Entities إلى نص عادي مناسب لتيليجرام و API
    """
    if not text:
        return ""

    # decode HTML entities: &quot; &#039; ...
    text = html.unescape(text)

    # normalize <br> to newline
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)

    # remove other html tags
    text = re.sub(r"<[^>]+>", "", text)

    # fix odd spaced tags that appear sometimes like: </ DeepSeek-V3.
    text = text.replace("</ ", "</")

    # collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def _extract_challenge_values(html_text: str) -> Tuple[bytes, bytes, bytes]:
    matches = re.findall(r'toNumbers\("([a-f0-9]+)"\)', html_text, flags=re.IGNORECASE)
    if len(matches) < 3:
        raise RuntimeError("Challenge values not found in HTML.")
    key = bytes.fromhex(matches[0])
    iv = bytes.fromhex(matches[1])
    data = bytes.fromhex(matches[2])
    return key, iv, data


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Android)"})

    r = s.get(HOME_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()

    key, iv, data = _extract_challenge_values(r.text)
    test_cookie = AES.new(key, AES.MODE_CBC, iv).decrypt(data).hex()
    s.cookies.set("__test", test_cookie, domain=COOKIE_DOMAIN)

    s.get(WARMUP_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    time.sleep(0.2)
    return s


def _get_session() -> requests.Session:
    global _session, _session_created_at
    with _lock:
        now = time.time()
        if _session is None or (now - _session_created_at) > SESSION_TTL_SECONDS:
            _session = _build_session()
            _session_created_at = now
        return _session


def _post_chat(session: requests.Session, model: str, question: str) -> requests.Response:
    payload = {"question": question, "model": model}

    log(f"DEBUG_PAYLOAD: {payload}")

    r = session.post(
        CHAT_URL,
        params={"i": "1"},
        data=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    # ensure utf-8 to reduce ��
    r.encoding = "utf-8"
    return r


def _parse_answer(html_text: str) -> str:
    # محاولة التقاط الرد من div
    m = re.search(
        r'<div class="response-content">(.*?)</div>',
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # لو رجعت صفحة HTML كاملة (فشل) نرجع فارغ
    return ""


def _extract_form_info(html_text: str) -> Dict[str, object]:
    """
    Debug: يحاول يقرأ أسماء حقول الفورم لو رجعت الصفحة الرئيسية بدل الرد.
    """
    info: Dict[str, object] = {"form_action": "", "field_names": [], "select_names": []}

    form = re.search(r"<form[^>]*>", html_text, flags=re.IGNORECASE)
    if form:
        form_tag = form.group(0)
        act = re.search(r'action="([^"]*)"', form_tag, flags=re.IGNORECASE)
        if act:
            info["form_action"] = act.group(1)

    fields = re.findall(r'<input[^>]+name="([^"]+)"', html_text, flags=re.IGNORECASE)
    selects = re.findall(r"<select[^>]+name=\"([^\"]+)\"", html_text, flags=re.IGNORECASE)

    info["field_names"] = sorted(list(set(fields)))
    info["select_names"] = sorted(list(set(selects)))
    return info


# -------------------------
# API endpoints
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatReq, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is not None and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # model required by the upstream site
    model = (req.model or "").strip() or DEFAULT_MODEL

    session = _get_session()

    try:
        r = _post_chat(session, model, question)
        r.raise_for_status()
    except Exception:
        # refresh session and retry once
        with _lock:
            global _session
            _session = None
        session = _get_session()
        r = _post_chat(session, model, question)
        r.raise_for_status()

    # Debug snippet
    if DEBUG:
        snippet = r.text[:500]
        log("DEBUG_RESPONSE_SNIPPET:\n" + snippet)

    answer_raw = _parse_answer(r.text)
    answer = clean_html_response(answer_raw)

    # لو رجعت الصفحة الرئيسية بدل الرد
    note = ""
    form_info = None
    if not answer:
        note = "Site returned main HTML page (not a chat response). Request format may differ or model may be invalid."
        form_info = _extract_form_info(r.text)
        if DEBUG:
            log(f"DEBUG_FORM_INFO: {form_info}")

    resp = {
        "model": model,
        "question": question,
        "answer": answer,
    }
    if note:
        resp["note"] = note
    if form_info:
        resp["debug_form_info"] = form_info

    return resp


# -------------------------
# Telegram webhook section
# -------------------------
async def tg_send(chat_id: int, text: str):
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(url, json=payload)


def get_chat_model(chat_id: int) -> str:
    return _chat_models.get(chat_id, DEFAULT_MODEL)


def set_chat_model(chat_id: int, model: str):
    _chat_models[chat_id] = model


@app.post("/tg/webhook")
async def tg_webhook(request: Request):
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN missing")

    update = await request.json()
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = (message.get("chat") or {}).get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}

    # Commands
    if text in ("/start", "/help"):
        await tg_send(
            chat_id,
            "أهلًا! 👋\n\n"
            "اكتب سؤالك وسأرد عليك.\n\n"
            "أوامر مفيدة:\n"
            "/models - عرض النماذج\n"
            "/model DeepSeek-V3 - اختيار نموذج\n"
            "/reset_model - رجوع للافتراضي\n"
            "/whoami - عرض النموذج الحالي"
        )
        return {"ok": True}

    if text == "/models":
        await tg_send(chat_id, "النماذج المتاحة:\n" + "\n".join(f"- {m}" for m in AVAILABLE_MODELS))
        return {"ok": True}

    if text == "/whoami":
        await tg_send(chat_id, f"النموذج الحالي: {get_chat_model(chat_id)}")
        return {"ok": True}

    if text == "/reset_model":
        set_chat_model(chat_id, DEFAULT_MODEL)
        await tg_send(chat_id, f"تم الرجوع إلى النموذج الافتراضي: {DEFAULT_MODEL}")
        return {"ok": True}

    if text.startswith("/model"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send(chat_id, "اكتب اسم النموذج بعد الأمر:\nمثال: /model DeepSeek-V3")
            return {"ok": True}

        chosen = parts[1].strip()
        if chosen not in AVAILABLE_MODELS:
            await tg_send(
                chat_id,
                "هذا النموذج غير موجود ضمن القائمة.\nاستخدم /models لعرض النماذج المتاحة."
            )
            return {"ok": True}

        set_chat_model(chat_id, chosen)
        await tg_send(chat_id, f"تم تعيين النموذج إلى: {chosen}")
        return {"ok": True}

    # Normal text => ask upstream
    model = get_chat_model(chat_id)
    try:
        req = ChatReq(question=text, model=model)
        res = chat(req, x_api_key=API_KEY)
        answer = res.get("answer") or "لا يوجد رد."

        # تنظيف احتياطي إضافي
        answer = clean_html_response(answer)
    except Exception:
        answer = "حصل خطأ أثناء المعالجة. جرّب مرة أخرى."

    # Telegram limit ~4096 chars
    if len(answer) > 4000:
        answer = answer[:4000] + "…"

    await tg_send(chat_id, answer)
    return {"ok": True}
