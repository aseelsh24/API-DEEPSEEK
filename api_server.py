import re
import time
import threading
from typing import Optional

import requests
from Crypto.Cipher import AES

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

BASE_URL = "https://asmodeus.free.nf"
HOME_URL = f"{BASE_URL}/"
WARMUP_URL = f"{BASE_URL}/index.php?i=1"
CHAT_URL = f"{BASE_URL}/deepseek.php"
COOKIE_DOMAIN = "asmodeus.free.nf"

API_KEY: Optional[str] = "20262025"

SESSION_TTL_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 60

app = FastAPI(title="Local Script API")

_lock = threading.Lock()
_session: Optional[requests.Session] = None
_session_created_at: float = 0.0


class ChatReq(BaseModel):
    model: str
    question: str


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
    return session.post(
        CHAT_URL,
        params={"i": "1"},
        data={"model": model, "question": question},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatReq, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is not None and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.model.strip():
        raise HTTPException(status_code=400, detail="model is required")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    session = _get_session()

    try:
        r = _post_chat(session, req.model, req.question)
        r.raise_for_status()
    except Exception:
        with _lock:
            global _session
            _session = None
        session = _get_session()
        r = _post_chat(session, req.model, req.question)
        r.raise_for_status()

    m = re.search(
        r'<div class="response-content">(.*?)</div>',
        r.text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    answer = m.group(1).strip() if m else ""

    return {
        "model": req.model,
        "question": req.question,
        "answer": answer,
    }
