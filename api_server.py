import os
import re
import json
import html
import time
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="DeepSeek Model Hub + Telegram Bot")

# =========================
# Configuration (ENV VARS)
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
# Your Render URL, e.g. https://api-deepseek-1.onrender.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

# DeepSeek / LLM backend (choose ONE approach)
# Option A) OpenAI-compatible endpoint (recommended)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()  # e.g. https://api.deepseek.com
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL_FALLBACK = os.getenv("LLM_MODEL_FALLBACK", "deepseek-chat").strip()

# If you already have your own internal router endpoint, set:
# INTERNAL_ROUTER_URL="http://127.0.0.1:10000/..." (or another service)
INTERNAL_ROUTER_URL = os.getenv("INTERNAL_ROUTER_URL", "").strip()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-V3").strip()

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

# =========================
# In-memory per-user settings (simple)
# For production, move to Redis/DB.
# =========================
user_model: Dict[int, str] = {}  # telegram_user_id -> model


# =========================
# Helpers
# =========================
def normalize_text(s: str) -> str:
    """Remove HTML tags, unescape entities, normalize whitespace."""
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</?[^>]+>", "", s)  # strip any HTML tags
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    return s


def clamp_message(s: str, limit: int = 3500) -> str:
    """Telegram message limit ~4096. Keep safe."""
    if len(s) <= limit:
        return s
    return s[: limit - 30].rstrip() + "\n\n...(truncated)"


async def tg_send(chat_id: int, text: str) -> None:
    """Send Telegram message (plain text: no parse_mode)."""
    if not TELEGRAM_BOT_TOKEN:
        return
    text = clamp_message(normalize_text(text))
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(url, json=payload)


def get_user_current_model(tg_user_id: int) -> str:
    m = user_model.get(tg_user_id) or DEFAULT_MODEL
    if m not in AVAILABLE_MODELS:
        m = DEFAULT_MODEL
    return m


def set_user_current_model(tg_user_id: int, model: str) -> bool:
    if model in AVAILABLE_MODELS:
        user_model[tg_user_id] = model
        return True
    return False


# =========================
# LLM Calling
# =========================
async def call_llm(model_name: str, prompt: str) -> str:
    """
    Returns plain text answer.
    Priority:
      1) INTERNAL_ROUTER_URL (if you have your own router)
      2) OpenAI-compatible endpoint (LLM_BASE_URL + LLM_API_KEY)
      3) Fallback: simple echo
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return "اكتب النص أو السؤال الذي تريدني أن أتعامل معه."

    # 1) Your internal router (expects: {model, question} -> {answer})
    if INTERNAL_ROUTER_URL:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    INTERNAL_ROUTER_URL,
                    json={"model": model_name, "question": prompt},
                )
                r.raise_for_status()
                data = r.json()
                ans = data.get("answer") or data.get("response") or ""
                return normalize_text(ans) or "لم أستطع توليد رد."
        except Exception:
            # continue to other option
            pass

    # 2) OpenAI-compatible (DeepSeek / others)
    if LLM_BASE_URL and LLM_API_KEY:
        # Many providers use /v1/chat/completions
        url = LLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": LLM_MODEL_FALLBACK,
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant. Selected model label: {model_name}."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return normalize_text(content) or "لم أستطع توليد رد."
        except Exception:
            return "حدث خطأ أثناء الاتصال بالنموذج. تأكد من LLM_BASE_URL و LLM_API_KEY."

    # 3) Fallback
    return f"✅ استلمت رسالتك: {prompt}\n\n(لم يتم ضبط مزود LLM بعد. ضع INTERNAL_ROUTER_URL أو LLM_BASE_URL/LLM_API_KEY)"


# =========================
# API Schemas
# =========================
class ChatRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    question: str = Field(default="")
    answer: Optional[str] = None  # optional (ignore if sent)


class ChatResponse(BaseModel):
    model: str
    question: str
    answer: str


# =========================
# Web UI (optional)
# =========================
@app.get("/", response_class=HTMLResponse)
async def home():
    options = "\n".join([f'<option value="{m}">{m}</option>' for m in AVAILABLE_MODELS])
    html_page = f"""
<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>مجمع نماذج DeepSeek</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto; margin: 24px; }}
    .card {{ max-width: 900px; margin: 0 auto; padding: 16px; border: 1px solid #ddd; border-radius: 12px; }}
    textarea {{ width: 100%; min-height: 120px; }}
    select, button, textarea {{ font-size: 16px; padding: 10px; border-radius: 10px; border: 1px solid #ccc; }}
    button {{ cursor: pointer; }}
    pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 12px; }}
    .row {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>مجمع نماذج DeepSeek</h2>
    <p>اختر نموذجًا ثم اكتب سؤالك.</p>

    <div class="row">
      <label>النموذج:</label>
      <select id="model">{options}</select>
      <button onclick="setDefault()">ضبط كافتراضي في الصفحة</button>
    </div>

    <div style="margin-top:12px;">
      <textarea id="q" placeholder="اكتب سؤالك هنا..."></textarea>
    </div>

    <div class="row" style="margin-top:12px;">
      <button onclick="send()">إرسال</button>
    </div>

    <h3 style="margin-top:16px;">الرد:</h3>
    <pre id="out"></pre>
  </div>

<script>
let defaultModel = "{DEFAULT_MODEL}";
document.getElementById("model").value = defaultModel;

function setDefault() {{
  defaultModel = document.getElementById("model").value;
}}

async function send() {{
  const model = document.getElementById("model").value;
  const question = document.getElementById("q").value;
  const out = document.getElementById("out");
  out.textContent = "جاري الإرسال...";
  const r = await fetch("/chat", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ model, question }})
  }});
  const data = await r.json();
  out.textContent = data.answer || JSON.stringify(data, null, 2);
}}
</script>
</body>
</html>
"""
    return HTMLResponse(html_page)


@app.get("/health")
async def health():
    return {"ok": True, "time": int(time.time())}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model = req.model.strip() if req.model else DEFAULT_MODEL
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL

    answer = await call_llm(model, req.question or "")
    answer = normalize_text(answer)  # IMPORTANT: no <br/>, no entities
    return ChatResponse(model=model, question=req.question or "", answer=answer)


# =========================
# Telegram Webhook
# =========================
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return JSONResponse({"ok": True, "ignored": True})

    chat_id = msg.get("chat", {}).get("id")
    from_user = msg.get("from", {}) or {}
    tg_user_id = from_user.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not tg_user_id:
        return JSONResponse({"ok": True})

    # Commands
    if text.startswith("/start"):
        await tg_send(chat_id,
            "أهلًا! أنا بوت مساعد يدعم عدة نماذج.\n\n"
            "اكتب أي سؤال وسأجيبك.\n"
            "الأوامر:\n"
            "/models\n"
            "/set_model DeepSeek-V3\n"
            "/current_model\n"
            "/reset_model\n"
            "/help"
        )
        return JSONResponse({"ok": True})

    if text.startswith("/help"):
        await tg_send(chat_id,
            "طريقة الاستخدام:\n"
            "- أرسل أي رسالة للرد مباشرة.\n\n"
            "الأوامر:\n"
            "/models - عرض النماذج\n"
            "/set_model <MODEL> - اختيار نموذج\n"
            "/current_model - عرض النموذج الحالي\n"
            "/reset_model - الرجوع للافتراضي\n\n"
            "أمثلة:\n"
            "/set_model DeepSeek-V3\n"
            "/translate ar en مرحبا\n"
        )
        return JSONResponse({"ok": True})

    if text.startswith("/models"):
        await tg_send(chat_id, "النماذج المتاحة:\n" + "\n".join(AVAILABLE_MODELS))
        return JSONResponse({"ok": True})

    if text.startswith("/current_model"):
        await tg_send(chat_id, f"النموذج الحالي: {get_user_current_model(tg_user_id)}")
        return JSONResponse({"ok": True})

    if text.startswith("/reset_model"):
        user_model.pop(tg_user_id, None)
        await tg_send(chat_id, f"تم الرجوع للنموذج الافتراضي: {DEFAULT_MODEL}")
        return JSONResponse({"ok": True})

    if text.startswith("/set_model"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send(chat_id, "اكتب اسم النموذج بعد الأمر.\nمثال: /set_model DeepSeek-V3")
            return JSONResponse({"ok": True})
        chosen = parts[1].strip()
        if set_user_current_model(tg_user_id, chosen):
            await tg_send(chat_id, f"تم اختيار النموذج: {chosen}")
        else:
            await tg_send(chat_id, "نموذج غير صحيح. استخدم /models لرؤية القائمة.")
        return JSONResponse({"ok": True})

    # Simple utility commands (optional)
    if text.startswith("/translate"):
        # Example: /translate ar en مرحبا
        parts = text.split(maxsplit=3)
        if len(parts) < 4:
            await tg_send(chat_id, "مثال: /translate ar en مرحبا")
            return JSONResponse({"ok": True})
        src, dst, content = parts[1], parts[2], parts[3]
        model = get_user_current_model(tg_user_id)
        prompt = f"Translate from {src} to {dst}. Return only the translation.\n\nText:\n{content}"
        ans = await call_llm(model, prompt)
        await tg_send(chat_id, ans)
        return JSONResponse({"ok": True})

    if text.startswith("/summarize"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send(chat_id, "اكتب النص بعد الأمر.\nمثال: /summarize نص طويل...")
            return JSONResponse({"ok": True})
        model = get_user_current_model(tg_user_id)
        prompt = f"Summarize the following text in 5 bullet points:\n\n{parts[1]}"
        ans = await call_llm(model, prompt)
        await tg_send(chat_id, ans)
        return JSONResponse({"ok": True})

    if text.startswith("/improve"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await tg_send(chat_id, "اكتب النص بعد الأمر.\nمثال: /improve هذا نص...")
            return JSONResponse({"ok": True})
        model = get_user_current_model(tg_user_id)
        prompt = f"Improve the following text (clear, correct, professional) and return only the improved version:\n\n{parts[1]}"
        ans = await call_llm(model, prompt)
        await tg_send(chat_id, ans)
        return JSONResponse({"ok": True})

    # Normal chat: use chosen model
    model = get_user_current_model(tg_user_id)
    ans = await call_llm(model, text)
    await tg_send(chat_id, ans)
    return JSONResponse({"ok": True})


# =========================
# Optional: Set webhook endpoint
# =========================
@app.post("/telegram/set_webhook")
async def set_webhook():
    if not TELEGRAM_BOT_TOKEN:
        return JSONResponse({"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"})
    if not PUBLIC_BASE_URL:
        return JSONResponse({"ok": False, "error": "Missing PUBLIC_BASE_URL (e.g. https://api-deepseek-1.onrender.com)"})

    webhook_url = PUBLIC_BASE_URL.rstrip("/") + "/telegram/webhook"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json={"url": webhook_url})
        return JSONResponse(r.json())
