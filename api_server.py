import os
import re
import time
import threading
from typing import Optional

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

API_KEY: Optional[str] = os.getenv("API_KEY", "20262025")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# خليها من نفس القائمة اللي طلعت لك
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-V3")

SESSION_TTL_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 60

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

    # هيدرز متصفح أساسية
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Android 13; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })

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


def _extract_models_from_html(html: str) -> list[str]:
    models = re.findall(r'<option\s+value="([^"]+)"', html, flags=re.IGNORECASE)
    out = []
    for m in models:
        m = m.strip()
        if m and m not in out:
            out.append(m)
    return out


def _extract_form_fields(html: str) -> dict:
    form_action = ""
    m_action = re.search(r"<form[^>]*action=['\"]([^'\"]+)['\"]", html, flags=re.IGNORECASE)
    if m_action:
        form_action = m_action.group(1)

    names = re.findall(
        r"<(input|textarea|select)[^>]*name=['\"]([^'\"]+)['\"]",
        html,
        flags=re.IGNORECASE,
    )

    select_names = []
    for tag, name in names:
        if tag.lower() == "select" and name not in select_names:
            select_names.append(name)

    return {
        "form_action": form_action,
        "field_names": [n[1] for n in names],
        "select_names": select_names,
    }


def _is_main_page(html: str) -> bool:
    return "<title>مجمع نماذج DeepSeek</title>" in html


def _extract_answer(html: str) -> str:
    # 1) الشكل الذي كنت تستخدمه
    m = re.search(r'<div class="response-content">(.*?)</div>', html, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) أحيانًا تكون النتيجة في عنصر آخر
    m2 = re.search(r'id=["\']response["\']\s*>\s*(.*?)<', html, flags=re.DOTALL | re.IGNORECASE)
    if m2:
        return m2.group(1).strip()

    return ""


def _post_chat(session: requests.Session, model: str, question: str, ajax: bool = False) -> requests.Response:
    payload = {"model": model, "question": question}
    print("DEBUG_PAYLOAD:", payload)

    headers = {
        # مهم جدًا لبعض المواقع
        "Referer": CHAT_URL,
        "Origin": BASE_URL,
    }

    if ajax:
        headers.update({
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "*/*",
        })

    return session.post(
        CHAT_URL,
        params={"i": "1"},
        data=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatReq, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is not None and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    model_to_use = (req.model or DEFAULT_MODEL).strip()
    if not model_to_use:
        model_to_use = DEFAULT_MODEL

    session = _get_session()

    # محاولة 1
    try:
        r = _post_chat(session, model_to_use, req.question, ajax=False)
        r.raise_for_status()
    except Exception:
        with _lock:
            global _session
            _session = None
        session = _get_session()
        r = _post_chat(session, model_to_use, req.question, ajax=False)
        r.raise_for_status()

    print("DEBUG_RESPONSE_SNIPPET:", r.text[:250])

    # إذا رجعت الصفحة الرئيسية → جرّب AJAX
    if _is_main_page(r.text):
        r2 = _post_chat(session, model_to_use, req.question, ajax=True)
        print("DEBUG_RESPONSE_SNIPPET_AJAX:", r2.text[:250])

        # لو AJAX جاب جواب
        answer2 = _extract_answer(r2.text)
        if answer2:
            return {"model": model_to_use, "question": req.question, "answer": answer2}

        # لو برضه رجعت Main Page
        if _is_main_page(r2.text):
            models = _extract_models_from_html(r2.text)
            info = _extract_form_fields(r2.text)
            print("DEBUG_FORM_INFO:", info)

            return {
                "model": model_to_use,
                "question": req.question,
                "answer": "",
                "note": "Still getting the main HTML page. The site may require a different endpoint (AJAX/fetch) inside JS or extra hidden fields.",
                "available_models": models[:50],
                "form_info": info,
            }

    # لو ليست main page: حاول استخراج جواب من المحاولة الأولى
    answer = _extract_answer(r.text)
    return {"model": model_to_use, "question": req.question, "answer": answer}


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

    try:
        update = await request.json()
    except Exception:
        return {"ok": True}

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

    try:
        req = ChatReq(question=text)
        res = chat(req, x_api_key=API_KEY)
        answer = res.get("answer") or res.get("note") or "لا يوجد رد."
    except Exception:
        answer = "حصل خطأ أثناء المعالجة. جرّب مرة أخرى."

    if len(answer) > 4000:
        answer = answer[:4000] + "…"

    await tg_send(chat_id, answer)
    return {"ok": True}
