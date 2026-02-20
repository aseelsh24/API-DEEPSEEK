import os
import re
import time
import threading
from typing import Optional, Dict

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
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-V3")  # خليته V3 لأنه شغال عندك

SESSION_TTL_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 60

# قائمة النماذج (حسب اللي ظهر معك)
AVAILABLE_MODELS = [
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

# تخزين نموذج كل مستخدم (حسب chat_id)
_user_models: Dict[int, str] = {}
_user_lock = threading.Lock()

app = FastAPI(title="Local Script API")

_lock = threading.Lock()
_session: Optional[requests.Session] = None
_session_created_at: float = 0.0


class ChatReq(BaseModel):
    question: str
    model: Optional[str] = None


def _extract_challenge_values(html: str) -> tuple[bytes, bytes, bytes]:
    matches = re.findall(r'toNumbers\("([a-f0-9]+)"\)', html, flags=re.IGNORECASE)
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
    # ✅ نرسل model دائمًا لأن الموقع غالبًا يحتاجه
    payload = {"question": question, "model": model}

    print("DEBUG_PAYLOAD:", payload, flush=True)

    return session.post(
        CHAT_URL,
        params={"i": "1"},
        data=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _extract_answer(html: str) -> str:
    # أحيانًا الرد يكون داخل div.response-content
    m = re.search(
        r'<div class="response-content">(.*?)</div>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return (m.group(1) or "").strip()

    # إذا رجع صفحة HTML كاملة، هذا يعني غالبًا خطأ في الفورم/الموديل
    if "<!DOCTYPE html" in html or "<html" in html.lower():
        print("DEBUG_RESPONSE_SNIPPET:\n", html[:600], flush=True)
        return ""

    return ""


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatReq, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is not None and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    # ✅ استخدم الموديل القادم من الطلب أو الافتراضي
    model = (req.model or "").strip() or DEFAULT_MODEL

    session = _get_session()

    try:
        r = _post_chat(session, model, req.question.strip())
        r.raise_for_status()
    except Exception:
        # إعادة بناء السيشن لو حصلت مشكلة
        with _lock:
            global _session
            _session = None
        session = _get_session()
        r = _post_chat(session, model, req.question.strip())
        r.raise_for_status()

    answer = _extract_answer(r.text)

    return {
        "model": model,
        "question": req.question.strip(),
        "answer": answer,
    }


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


def _get_user_model(chat_id: int) -> str:
    with _user_lock:
        return _user_models.get(chat_id, DEFAULT_MODEL)


def _set_user_model(chat_id: int, model: str):
    with _user_lock:
        _user_models[chat_id] = model


def _reset_user_model(chat_id: int):
    with _user_lock:
        if chat_id in _user_models:
            del _user_models[chat_id]


def _format_models_list() -> str:
    # عرض مرتب
    lines = ["📌 النماذج المتاحة:"]
    for m in AVAILABLE_MODELS:
        lines.append(f"- {m}")
    lines.append("\n✅ للاختيار: /model DeepSeek-V3")
    lines.append("✅ لمعرفة الحالي: /model")
    lines.append("✅ للرجوع للافتراضي: /reset_model")
    return "\n".join(lines)


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

    # أوامر عامة
    if text in ("/start", "/help"):
        current = _get_user_model(chat_id)
        await tg_send(
            chat_id,
            "أهلًا! 👋\n"
            "اكتب سؤالك وسأرد عليك.\n\n"
            f"🔧 النموذج الحالي: {current}\n"
            "📌 عرض النماذج: /models\n"
            "✅ تغيير النموذج: /model DeepSeek-V3\n"
            "♻️ رجوع للافتراضي: /reset_model"
        )
        return {"ok": True}

    if text == "/models":
        await tg_send(chat_id, _format_models_list())
        return {"ok": True}

    if text == "/model":
        await tg_send(chat_id, f"🔧 النموذج الحالي لديك: {_get_user_model(chat_id)}")
        return {"ok": True}

    if text.startswith("/model "):
        desired = text.replace("/model", "", 1).strip()
        if desired not in AVAILABLE_MODELS:
            await tg_send(
                chat_id,
                "❌ اسم النموذج غير صحيح.\n\n"
                "استخدم /models لعرض الأسماء المتاحة."
            )
            return {"ok": True}

        _set_user_model(chat_id, desired)
        await tg_send(chat_id, f"✅ تم تغيير النموذج إلى: {desired}")
        return {"ok": True}

    if text == "/reset_model":
        _reset_user_model(chat_id)
        await tg_send(chat_id, f"♻️ تم الرجوع إلى النموذج الافتراضي: {DEFAULT_MODEL}")
        return {"ok": True}

    # أي رسالة أخرى: نرسلها للـ /chat باستخدام نموذج المستخدم
    try:
        model = _get_user_model(chat_id)
        req = ChatReq(question=text, model=model)
        res = chat(req, x_api_key=API_KEY)
        answer = (res.get("answer") or "").strip() or "لا يوجد رد."
    except Exception:
        answer = "حصل خطأ أثناء المعالجة. جرّب مرة أخرى."

    # حد تيليجرام ~4096 حرف
    if len(answer) > 4000:
        answer = answer[:4000] + "…"

    await tg_send(chat_id, answer)
    return {"ok": True}
