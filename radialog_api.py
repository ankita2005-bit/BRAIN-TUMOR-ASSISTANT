
import base64
import io
import os
import subprocess
import uuid
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image


BACKEND = os.getenv("RADIALOG_BACKEND", "ollama_llava")


app = FastAPI(title="RaDialog Local API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CONVS: Dict[str, Dict[str, Any]] = {}

def _pil_from_bytes(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _b64_from_bytes(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def _ensure_conv(conv_id: str) -> Dict[str, Any]:
    if conv_id not in CONVS:
        CONVS[conv_id] = {"history": [], "image_b64": None}
    return CONVS[conv_id]


def _run_llava_with_ollama(image: Image.Image, prompt: str) -> str:
    """Calls local Ollama (llava) with image + prompt."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")
        full_prompt = f"<image:{tmp.name}>\n{prompt}"
        try:
            res = subprocess.run(
                ["ollama", "run", "llava"],
                input=full_prompt,
                text=True,
                capture_output=True,
                check=True,
            )
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"⚠️ LLaVA Error: {e.stderr or str(e)}"



def _first_turn_generate(image_b64: str, findings: str) -> str:
    """FIRST turn: generate the initial radiology report."""
    prompt = f"""
You are a radiologist. This brain MRI shows a {findings}.
Generate a concise, structured radiology report with:
- Technique (assume standard MRI brain)
- Findings (one paragraph, radiology style; avoid enumeration)
- Impression (bullet points, 1–3 lines, most important first)

Important:
- Use correct neuroanatomical left/right conventions.
- Do not mention you are an AI. Do not invent history beyond "{findings}".
"""
    if BACKEND == "ollama_llava":
        img = _pil_from_bytes(base64.b64decode(image_b64))
        return _run_llava_with_ollama(img, prompt)
    else:
        return "⚠️ RaDialog backend not wired yet. Use RADIALOG_BACKEND=ollama_llava."


def _chat_turn(conv: Dict[str, Any], user_msg: str) -> str:
    """FOLLOW-UP turns: refine or answer about report."""
    system_msg = (
        "You are a board-certified radiologist assistant. "
        "Refine the report based on user instructions while keeping radiology style. "
        "Be precise and concise."
    )

    last_assistant = ""
    for m in reversed(conv["history"]):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    edit_prompt = f"""
{system_msg}

Current report:

<<<REPORT_START>>>
{last_assistant or "(no report yet)"}
<<<REPORT_END>>>

User request:
{user_msg}

Return only the revised report (or direct answer if it's a question).
"""
    if BACKEND == "ollama_llava":
        return _run_llava_with_ollama(
            _pil_from_bytes(base64.b64decode(conv["image_b64"])), edit_prompt
        )
    else:
        return "⚠️ RaDialog backend not wired yet."


def _qa_turn(conv: Dict[str, Any], question: str) -> str:
    """Answer doctor’s questions about the report + MRI image."""
    last_report = ""
    for m in reversed(conv["history"]):
        if m["role"] == "assistant":
            last_report = m["content"]
            break

    qa_prompt = f"""
You are a radiology expert.
Here is the most recent MRI report:

<<<REPORT>>>
{last_report}
<<<END REPORT>>>

Doctor’s question:
{question}

Answer clearly and concisely, using information from both the report and the MRI image.
If uncertain, state limitations.
"""
    if BACKEND == "ollama_llava":
        return _run_llava_with_ollama(
            _pil_from_bytes(base64.b64decode(conv["image_b64"])), qa_prompt
        )
    else:
        return "⚠️ RaDialog backend not wired yet."




@app.post("/generate_report")
async def generate_report(image: UploadFile = File(...), findings: str = Form("brain tumor")):
    try:
        conv_id = str(uuid.uuid4())
        conv = _ensure_conv(conv_id)
        img_bytes = await image.read()
        img_b64 = _b64_from_bytes(img_bytes)
        conv["image_b64"] = img_b64

        conv["history"].append({"role": "system", "content": "You assist with radiology report generation."})
        report = _first_turn_generate(img_b64, findings)

        conv["history"].append({"role": "user", "content": f"Findings: {findings}"})
        conv["history"].append({"role": "assistant", "content": report})

        return JSONResponse({"conversation_id": conv_id, "report": report})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat")
async def chat(conversation_id: str = Form(...), message: str = Form(...)):
    try:
        conv = _ensure_conv(conversation_id)
        conv["history"].append({"role": "user", "content": message})
        reply = _chat_turn(conv, message)
        conv["history"].append({"role": "assistant", "content": reply})
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/report_qa")
async def report_qa(conversation_id: str = Form(...), question: str = Form(...)):
    try:
        conv = _ensure_conv(conversation_id)
        conv["history"].append({"role": "user", "content": question})
        reply = _qa_turn(conv, question)
        conv["history"].append({"role": "assistant", "content": reply})
        return JSONResponse({"answer": reply})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
