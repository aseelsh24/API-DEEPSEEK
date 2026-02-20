import os
import re
import time
import threading
import html
from typing import Optional, List, Dict, Any, Tuple

import requests
import httpx
from Crypto.Cipher import AES

from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel

BASE_URL = "https://asmodeus.free.nf"
HOME_URL = f"{BASE_URL}/"
WARMUP_URL = f"{BASE_URL}/index.php?i=1"
CHAT_URL = f"{BASE_URL}/deepseek.php"
COOKIE_DOMAIN = "asmodeus.free.nf"

# الأفضل تخزنها في Render Env بدل الكود
API_KEY: Optional[str] = os.getenv("API_KEY", "20262025")

# تيليجرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-V3")

SESSION_TTL_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 60

# Debug (اختياري)
DEBUG = os.getenv("DEBUG", "0") == "1"

app = FastAPI(title="Local Script API")

_lock = threading.Lock()
_session: Optional[requests.Session] = None
_session_created_at: float = 0.0


class ChatReq(BaseModel):
    question: str
    model: Optional[str] = None


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
    # ✅ الموقع يحتاج model + question (وأنت لاحظت هذا فعليًا)
    payload = {
        "model": model,
        "question": question,
    }

    if DEBUG:
        print("DEBUG_PAYLOAD:", payload)

    return session.post(
        CHAT_URL,
        params={"i": "1"},
        data=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _clean_answer(text: str) -> str:
    """
    - يفك ترميز HTML entities مثل &#039; و &quot;
    - يحول <br> إلى أسطر جديدة
    - يشيل أي HTML tags متبقية
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", "", text)  # remove any remaining tags
    return text.strip()


def _extract_answer_from_html(page_html: str) -> str:
    m = re.search(
        r'<div class="response-content">(.*?)</div>',
        page_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    raw = m.group(1).strip() if m else ""
    return _clean_answer(raw)


def _extract_available_models(page_html: str) -> List[str]:
    # يحاول استخراج الخيارات من <select name="model"> ... <option>...</option>
    models = re.findall(r"<option[^>]*>\s*([^<]+?)\s*</option>", page_html, flags=re.IGNORECASE)
    cleaned = []
    for m in models:
        val = m.strip()
        if val and val.lower() not in ("اختر", "choose", "select", "model"):
            cleaned.append(val)
    # إزالة التكرار مع الحفاظ على الترتيب
    seen = set()
    unique = []
    for x in cleaned:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatReq, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is not None and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    model = (req.model or "").strip() or DEFAULT_MODEL

    session = _get_session()

    try:
        r = _post_chat(session, model, req.question.strip())
        r.raise_for_status()
    except Exception:
        # إعادة بناء السيشن عند أي فشل
        with _lock:
            global _session
            _session = None
        session = _get_session()
        r = _post_chat(session, model, req.question.strip())
        r.raise_for_status()

    # إذا رجع صفحة رئيسية بدل نتيجة (يحصل لو format غلط أو model غلط)
    answer = _extract_answer_from_html(r.text)

    # لو ما قدرنا نطلع answer، رجّع ملاحظة + موديلات متاحة (لو قدرنا نستخرجها)
    resp: Dict[str, Any] = {
        "model": model,
        "question": req.question,
        "answer": answer,
    }

    if not answer:
        # Debug snippet في اللوج (اختياري)
        if DEBUG:
            print("DEBUG_RESPONSE_SNIPPET:\n", r.text[:500])

        resp["note"] = "Site returned main HTML page (not a chat response). Likely model is required or request format differs."
        resp["available_models"] = _extract_available_models(r.text)

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

    if text in ("/start", "/help"):
        await tg_send(chat_id, "أهلًا! اكتب سؤالك وسأرد عليك.")
        return {"ok": True}

    # نستخدم نفس منطق /chat (بدون HTTP داخلي)
    try:
        req = ChatReq(model=DEFAULT_MODEL, question=text)
        res = chat(req, x_api_key=API_KEY)
        answer = (res.get("answer") or "").strip()

        if not answer:
            note = res.get("note") or "لا يوجد رد."
            models = res.get("available_models") or []
            if models:
                answer = f"{note}\n\nAvailable models:\n- " + "\n- ".join(models[:30])
            else:
                answer = note

    except Exception:
        answer = "حصل خطأ أثناء المعالجة. جرّب مرة أخرى."

    # حد تيليجرام ~4096 حرف
    if len(answer) > 4000:
        answer = answer[:4000] + "…"

    await tg_send(chat_id, answer)
    return {"ok": True}
