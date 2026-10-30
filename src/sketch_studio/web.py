"""Local web studio (BRD §9: localhost UI) — mind map editor + character gallery."""

import asyncio
import io
import json
import re
import socket
import sys
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import config, db, llm, service
from .comfy import EXPRESSIONS, STYLE_PRESETS, ComfyError

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Sketch-to-2D Studio")
app.mount("/storage", StaticFiles(directory=config.STORAGE), name="storage")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/projects")
def projects():
    return db.list_projects()


@app.get("/api/projects/{pid}")
def project(pid: str):
    p = db.get_project(pid)
    if not p:
        raise HTTPException(404, "project not found")
    graph = db.latest_graph(pid)
    return {
        "project": p,
        "graph_version": graph[0] if graph else None,
        "graph": graph[1] if graph else None,
        "profile": db.latest_profile(pid),
        "assets": db.list_assets(pid),
        "jobs": db.list_jobs(pid),
    }


@app.put("/api/projects/{pid}/mindmap")
def save_mindmap(pid: str, graph: dict = Body(...)):
    if not db.get_project(pid):
        raise HTTPException(404, "project not found")
    try:
        return {"version": service.save_graph_from_web(pid, graph)}
    except (ValidationError, ValueError) as e:
        raise HTTPException(422, str(e))


@app.post("/api/jobs/{jid}/refresh")
async def refresh_job(jid: str):
    """Lets the gallery pull finished ComfyUI results without asking Claude."""
    try:
        job = await service.poll_job(jid, wait_seconds=0)
    except Exception as e:
        raise HTTPException(502, str(e))
    return {"status": job["status"], "error": job["error"]}


# ---------- web-only flow (self-hosted LLM) ----------

_tasks: set[asyncio.Task] = set()


def _background(coro) -> None:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _call(coro):
    """Run a service call, mapping expected failures to readable HTTP errors."""
    try:
        return await coro
    except (llm.LLMError, ComfyError) as e:
        raise HTTPException(502, str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))


@app.get("/api/settings")
async def get_settings():
    out = {"comfy_url": db.get_setting("comfy_url", config.COMFY_URL),
           "llm_url": db.get_setting("llm_url", config.LLM_URL),
           "llm_model": db.get_setting("llm_model", config.LLM_MODEL),
           "styles": list(STYLE_PRESETS)}
    try:
        out["comfy"] = await service.client().status()
    except Exception as e:
        out["comfy"] = {"error": str(e)}
    try:
        out["llm"] = {"models": await llm.client().models()}
    except Exception as e:
        out["llm"] = {"error": str(e)}
    return out


class Settings(BaseModel):
    comfy_url: str = ""
    llm_url: str = ""
    llm_model: str = ""


@app.put("/api/settings")
async def put_settings(s: Settings):
    for k, v in s.model_dump().items():
        db.set_setting(k, v.strip().rstrip("/"))
    return await get_settings()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), analyze: bool = True):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.IMAGE_EXTS:
        raise HTTPException(422, f"Định dạng {suffix or '?'} không hỗ trợ")
    stem = re.sub(r"[^\w.-]+", "_", Path(file.filename).stem)[:40] or "sketch"
    dest = config.INBOX / f"{stem}_{uuid.uuid4().hex[:4]}{suffix}"
    dest.write_bytes(await file.read())
    pid = service.open_sketch(dest.name)["project_id"]
    if analyze:
        db.update_project(pid, status="analyzing")
        _background(service.analyze_project(pid))
    return {"project_id": pid}


class AnalyzeReq(BaseModel):
    category: str | None = None


@app.post("/api/projects/{pid}/analyze")
async def analyze(pid: str, req: AnalyzeReq):
    if not db.get_project(pid):
        raise HTTPException(404, "project not found")
    db.update_project(pid, status="analyzing")
    _background(service.analyze_project(pid, req.category))
    return {"status": "analyzing"}


class EditReq(BaseModel):
    instruction_vi: str


@app.post("/api/projects/{pid}/mindmap/edit")
async def edit_mindmap(pid: str, req: EditReq):
    return {"version": await _call(service.edit_mindmap(pid, req.instruction_vi))}


@app.post("/api/projects/{pid}/icons")
async def icons(pid: str):
    return await _call(service.auto_icons(pid))


@app.post("/api/projects/{pid}/concept")
async def concept(pid: str):
    await _call(service.concept_to_character(pid))
    return {"ok": True}


class GenerateReq(BaseModel):
    style: str = "anime"
    fidelity: str | None = None
    batch_size: int = 4
    extra_vi: str = ""
    transparent: bool = True


@app.post("/api/projects/{pid}/generate")
async def generate(pid: str, req: GenerateReq):
    extra = ""
    if req.extra_vi.strip():
        extra = (await _call(llm.translate_request(req.extra_vi, db.latest_profile(pid)))).prompt_delta_en
    return await _call(service.generate_character(pid, req.style, req.batch_size, None, req.fidelity, extra,
                                                  transparent=req.transparent))


class VariationReq(BaseModel):
    request_vi: str = ""
    prompt_delta_en: str = ""
    strength: float = 0.55
    keep_pose: bool = True
    batch_size: int = 2
    face_lock: float | None = None  # None = auto, 0 = off


@app.post("/api/assets/{aid}/variation")
async def variation(aid: str, req: VariationReq):
    if req.prompt_delta_en:  # suggestion chip: already an English prompt, no LLM call
        return await _call(service.generate_variation(aid, req.prompt_delta_en, req.strength, req.keep_pose,
                                                      req.batch_size, face_lock=req.face_lock))
    return await _call(service.variation_from_request(aid, req.request_vi, req.batch_size, face_lock=req.face_lock))


@app.get("/api/expressions")
def expressions():
    return [{"key": k, "label_vi": v[0]} for k, v in EXPRESSIONS.items()]


class ExpressionsReq(BaseModel):
    emotions: list[str] = []
    custom_vi: list[str] = []  # free-text emotions, translated by the LLM
    framing: str = "bust"
    strength: float | None = None
    face_lock: float | None = None


@app.post("/api/assets/{aid}/expressions")
async def make_expressions(aid: str, req: ExpressionsReq):
    asset = db.get_asset(aid)
    if not asset:
        raise HTTPException(404, "asset not found")
    profile = db.latest_profile(asset["project_id"])
    custom = []
    for text in (t.strip() for t in req.custom_vi):
        if text:
            custom.append((text, (await _call(llm.translate_request(f"biểu cảm: {text}", profile))).prompt_delta_en))
    return await _call(service.generate_expressions(aid, req.emotions, custom, req.framing, req.strength,
                                                    face_lock=req.face_lock))


@app.get("/api/jobs/{jid}/zip")
def job_zip(jid: str):
    import zipfile

    job = db.get_job(jid)
    if not job:
        raise HTTPException(404, "job not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for a in db.list_assets(job["project_id"], jid):
            label = (json.loads(a["meta_json"] or "{}").get("label_vi") or Path(a["path"]).stem).replace("/", "_")
            z.write(config.STORAGE / a["path"], f"{label}_{a['id']}.png")
    headers = {"Content-Disposition": f'attachment; filename="{jid}.zip"'}
    return Response(buf.getvalue(), media_type="application/zip", headers=headers)


@app.post("/api/assets/{aid}/sticker")
async def sticker(aid: str):
    try:
        r = await asyncio.to_thread(service.remove_background, aid)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))
    return {"asset_id": r["asset_id"]}


def port_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex((config.WEB_HOST, config.WEB_PORT)) != 0


def start_background() -> bool:
    """Serve the UI from inside the MCP process. Logs go to stderr: stdout is the MCP stdio channel."""
    if not port_free():
        print(f"[studio] port {config.WEB_PORT} busy — assuming another studio instance serves the UI", file=sys.stderr)
        return False
    cfg = uvicorn.Config(app, host=config.WEB_HOST, port=config.WEB_PORT, log_config=None, access_log=False)
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    return True


def main() -> None:
    db.init()
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)
