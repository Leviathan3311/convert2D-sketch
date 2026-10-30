"""Orchestration shared by the MCP tools and the web API."""

import asyncio
import time
from pathlib import Path

from . import comfy, config, db, preprocess
from .schemas import MindMapGraphStructure, UniversalSketchAnalysisResponse, validate_graph

ASSET_TYPE = {"character": "candidate", "variation": "variation", "icons": "icon", "expressions": "expression"}


def rel(path: Path) -> str:
    return path.relative_to(config.STORAGE).as_posix()


def url(rel_path: str) -> str:
    return f"/storage/{rel_path}"


def project_url(pid: str) -> str:
    return f"{config.WEB_URL}/#/p/{pid}"


def client() -> comfy.Comfy:
    return comfy.Comfy(db.get_setting("comfy_url", config.COMFY_URL))


def resolve_sketch(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if not p.is_absolute():
        p = config.INBOX / p
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy ảnh {p}. Bỏ ảnh vào {config.INBOX} hoặc đưa đường dẫn tuyệt đối.")
    if p.suffix.lower() not in config.IMAGE_EXTS:
        raise ValueError(f"Định dạng {p.suffix} không hỗ trợ")
    return p


def list_sketches() -> list[str]:
    return sorted(p.name for p in config.INBOX.iterdir() if p.suffix.lower() in config.IMAGE_EXTS)


# ---------- sketch intake & VLM router ----------

def open_sketch(name_or_path: str) -> dict:
    src = resolve_sketch(name_or_path)
    pid = db.new_id("prj")
    out = preprocess.process(src, config.STORAGE / pid)
    db.create_project(pid, src.stem, rel(out["raw"]), rel(out["clean"]), rel(out["control"]), out["aspect"])
    return {"project_id": pid, "raw_path": out["raw"], "aspect": out["aspect"], "url": project_url(pid)}


def submit_analysis(pid: str, analysis: UniversalSketchAnalysisResponse) -> dict:
    project = db.get_project(pid)
    if not project:
        raise ValueError(f"Không có project {pid}")
    cat = analysis.detected_input_category
    if cat == "mindmap_or_flowchart_diagram" and not analysis.mindmap_graph:
        raise ValueError("Mind map cần mindmap_graph")
    if cat in ("character_or_object_sketch", "character_concept_mindmap") and not analysis.character_profile:
        raise ValueError("Nhánh nhân vật cần character_profile")

    result: dict = {"project_id": pid, "category": cat, "url": project_url(pid)}
    if analysis.mindmap_graph:
        graph = validate_graph(analysis.mindmap_graph)
        result["mindmap_version"] = db.save_graph(pid, graph.model_dump(), source="claude")
        result["mermaid_code"] = graph.mermaid_code
    if analysis.character_profile:
        db.save_profile(pid, analysis.character_profile.model_dump())
    project_type = "mindmap_2d" if cat == "mindmap_or_flowchart_diagram" else "character_2d"
    if project["project_type"] == "hybrid_sketchnote" and cat == "mindmap_or_flowchart_diagram":
        project_type = "hybrid_sketchnote"
    db.update_project(pid, input_category=cat, project_type=project_type, status="analyzed")
    return result


def save_graph_from_web(pid: str, graph: dict) -> int:
    g = validate_graph(MindMapGraphStructure.model_validate(graph))
    return db.save_graph(pid, g.model_dump(), source="web")


# ---------- ComfyUI jobs ----------

async def _face_ref(c: comfy.Comfy, asset_path: Path, face_lock: float | None, default: float) -> tuple[str | None, float]:
    """Upload a face crop for IP-Adapter. face_lock None = auto (default weight if Colab has IP-Adapter), 0 = off."""
    if face_lock == 0:
        return None, 0.0
    if not await c.has_face_lock():
        if face_lock is None:
            return None, 0.0
        raise ValueError("Colab chưa có IP-Adapter (khoá mặt): chạy lại notebook mới, hoặc tắt khoá mặt")
    weight = default if face_lock is None else face_lock
    return await c.upload(preprocess.flat_png(asset_path, face=True, size=512)), weight


async def _queue(pid: str, kind: str, wf: dict, params: dict) -> dict:
    numbered, ids = comfy.numbered(wf)
    prompt_id = await client().queue(numbered)
    jid = db.create_job(pid, kind, "comfy_gpu", prompt_id, {**params, "node_ids": ids})
    return {"job_id": jid, "status": "running", "kind": kind}


async def generate_character(
    pid: str, style: str = "anime", batch_size: int = 4, seed: int | None = None,
    fidelity: str | None = None, extra_prompt_en: str = "", hires: bool = True, transparent: bool = True,
) -> dict:
    project = db.get_project(pid)
    profile = db.latest_profile(pid) if project else None
    if not profile:
        raise ValueError("Project chưa có character_profile — gọi submit_analysis trước")
    if style not in comfy.STYLE_PRESETS:
        raise ValueError(f"style phải là một trong {list(comfy.STYLE_PRESETS)}")
    fidelity = fidelity or profile["recommended_fidelity_mode"]
    use_sketch = project["input_category"] == "character_or_object_sketch"
    c = client()
    hint = await c.upload(config.STORAGE / project["sketch_control"]) if use_sketch else None
    size = preprocess.SIZES[project["aspect"] if use_sketch else "portrait"]
    seed = comfy.seed_or_random(seed)
    positive = comfy.character_prompt(profile, style, extra_prompt_en)
    wf = comfy.character_workflow(positive, size, batch_size, seed, hint, fidelity, hires)
    params = {"style": style, "seed": seed, "fidelity": fidelity, "prompt": positive, "sketch_control": use_sketch,
              "transparent": transparent}
    return {**await _queue(pid, "character", wf, params), "prompt": positive, "seed": seed}


async def generate_variation(
    asset_id: str, prompt_delta_en: str, strength: float = 0.55, keep_pose: bool = True,
    batch_size: int = 2, seed: int | None = None, face_lock: float | None = None,
    colored_items: list[tuple[str, str]] = (),
) -> dict:
    asset = db.get_asset(asset_id)
    if not asset:
        raise ValueError(f"Không có asset {asset_id}")
    pid = asset["project_id"]
    project, profile = db.get_project(pid), db.latest_profile(pid)
    parent_job = db.get_job(asset["job_id"]) if asset["job_id"] else None
    style = parent_job["params"].get("style", "anime") if parent_job else "anime"
    transparent = parent_job["params"].get("transparent", True) if parent_job else True
    c = client()
    src = await c.upload(preprocess.flat_png(config.STORAGE / asset["path"]))
    face, face_weight = await _face_ref(c, config.STORAGE / asset["path"], face_lock, 0.7)
    use_pose = keep_pose and project["input_category"] == "character_or_object_sketch"
    hint = await c.upload(config.STORAGE / project["sketch_control"]) if use_pose else None
    seed = comfy.seed_or_random(seed)
    delta, chunks, negative_extra = comfy.bind_colors(prompt_delta_en, list(colored_items))
    positive = comfy.character_prompt(profile, style, focus=delta, focus_weight=1.3)
    if chunks:
        positive = f"{positive} BREAK {chunks}"
    wf = comfy.variation_workflow(positive, src, strength, batch_size, seed, hint, face_image=face,
                                  face_weight=face_weight, negative_extra=negative_extra)
    params = {"style": style, "seed": seed, "prompt": positive, "parent_asset_id": asset_id, "strength": strength,
              "transparent": transparent, "face_lock": face_weight}
    return {**await _queue(pid, "variation", wf, params), "prompt": positive, "seed": seed, "face_lock": face_weight}


async def generate_expressions(
    asset_id: str, emotions: list[str] | None = None, custom: list[tuple[str, str]] = (),
    framing: str = "bust", strength: float | None = None, seed: int | None = None, face_lock: float | None = None,
) -> dict:
    """Expression pack from an approved image: same seed + same source latent, only the emotion prompt changes.
    custom = [(label_vi, prompt_en)] for emotions outside the presets."""
    asset = db.get_asset(asset_id)
    if not asset:
        raise ValueError(f"Không có asset {asset_id}")
    emotions = emotions if emotions is not None else list(comfy.EXPRESSIONS)[:8]
    unknown = [e for e in emotions if e not in comfy.EXPRESSIONS]
    if unknown:
        raise ValueError(f"Cảm xúc không có trong danh sách: {unknown}. Dùng custom cho cảm xúc khác.")
    labels = {e: comfy.EXPRESSIONS[e][0] for e in emotions}
    prompts = {e: comfy.EXPRESSIONS[e][1] for e in emotions}
    for i, (label, prompt) in enumerate(custom, 1):
        labels[f"custom{i}"], prompts[f"custom{i}"] = label, prompt
    if not prompts:
        raise ValueError("Chọn ít nhất một cảm xúc")
    if framing not in ("bust", "full"):
        raise ValueError("framing phải là bust (chân dung) hoặc full (toàn thân)")
    pid = asset["project_id"]
    profile = db.latest_profile(pid)
    parent_job = db.get_job(asset["job_id"]) if asset["job_id"] else None
    style = parent_job["params"].get("style", "anime") if parent_job else "anime"
    suffix = comfy.BUST_SUFFIX if framing == "bust" else comfy.BASE_SUFFIX
    # emotion first and up-weighted: with img2img + face lock a plain mention barely moves the face
    full = {k: comfy.character_prompt(profile, style, suffix=suffix, focus=p, focus_weight=1.3) for k, p in prompts.items()}
    c = client()
    src = await c.upload(preprocess.flat_png(config.STORAGE / asset["path"], bust=framing == "bust"))
    face, face_weight = await _face_ref(c, config.STORAGE / asset["path"], face_lock, 0.65)
    seed = comfy.seed_or_random(seed)
    # with the face locked by IP-Adapter, img2img can move further so the emotion actually shows
    strength = strength or ((0.68 if face else 0.55) if framing == "bust" else (0.6 if face else 0.5))
    wf = comfy.expressions_workflow(full, src, strength, seed, face_image=face, face_weight=face_weight)
    params = {"style": style, "seed": seed, "parent_asset_id": asset_id, "framing": framing, "strength": strength,
              "labels": labels, "prompts": full, "transparent": True, "face_lock": face_weight}
    return {**await _queue(pid, "expressions", wf, params), "emotions": labels, "seed": seed, "face_lock": face_weight}


async def generate_icons(pid: str, node_ids: list[str] | None = None) -> dict:
    latest = db.latest_graph(pid)
    if not latest:
        raise ValueError("Project chưa có mind map")
    nodes = [n for n in latest[1]["nodes"] if n.get("icon_prompt_en") and (not node_ids or n["node_id"] in node_ids)]
    if not nodes:
        raise ValueError("Không node nào có icon_prompt_en — gửi lại submit_analysis với icon_prompt_en cho các nhánh chính")
    seed = comfy.seed_or_random(None)
    wf = comfy.icons_workflow({n["node_id"]: n["icon_prompt_en"] for n in nodes}, seed)
    db.update_project(pid, project_type="hybrid_sketchnote")
    return await _queue(pid, "icons", wf, {"seed": seed, "nodes": [n["node_id"] for n in nodes]})


def _gpu_seconds(entry: dict) -> float | None:
    ts = {m[0]: m[1].get("timestamp") for m in entry.get("status", {}).get("messages", []) if isinstance(m, list)}
    if ts.get("execution_start") and ts.get("execution_success"):
        return (ts["execution_success"] - ts["execution_start"]) / 1000
    return None


async def _collect(job: dict, entry: dict) -> None:
    pid, kind, params = job["project_id"], job["kind"], job["params"]
    out_dir = config.STORAGE / pid / job["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    c = client()
    icon_urls: dict[str, str] = {}
    for name, num in params["node_ids"].items():
        if not name.startswith("save"):
            continue
        tag = name.removeprefix("save_") if name != "save" else None  # mind-map node id or emotion key
        for i, img in enumerate(entry.get("outputs", {}).get(num, {}).get("images", [])):
            path = out_dir / f"{tag or kind}_{i}.png"
            data = await c.view(img)
            if params.get("transparent"):
                data = await asyncio.to_thread(preprocess.cutout, data)
            path.write_bytes(data)
            meta = {"batch_index": i, "style": params.get("style"),
                    "prompt": params.get("prompts", {}).get(tag) or params.get("prompt")}
            if kind == "icons":
                meta["mind_node"] = tag
                icon_urls[tag] = url(rel(path))
            if kind == "expressions":
                meta.update(emotion=tag, label_vi=params["labels"].get(tag))
            db.add_asset(pid, job["id"], params.get("parent_asset_id"), ASSET_TYPE[kind], rel(path), params.get("seed"), meta)
    if icon_urls:
        version, graph = db.latest_graph(pid)
        for n in graph["nodes"]:
            n["icon_url"] = icon_urls.get(n["node_id"], n.get("icon_url"))
        db.save_graph(pid, graph, source="icons")
    with db.conn() as conn:
        conn.execute("UPDATE generation_jobs SET gpu_seconds=? WHERE id=?", (_gpu_seconds(entry), job["id"]))


async def poll_job(jid: str, wait_seconds: float = 45) -> dict:
    deadline = time.monotonic() + wait_seconds
    while True:
        job = db.get_job(jid)
        if not job:
            raise ValueError(f"Không có job {jid}")
        if job["status"] in ("done", "error"):
            return {**job, "assets": db.list_assets(job["project_id"], jid)}
        entry = await client().history(job["comfy_prompt_id"])
        status = (entry or {}).get("status", {})
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
            db.finish_job(jid, "error", str(msgs[-1][1] if msgs else status)[:2000])
            continue
        if entry:  # ComfyUI only writes history once the prompt has finished
            if db.claim_job(jid):
                try:
                    await _collect(job, entry)
                    db.finish_job(jid, "done")
                except Exception as e:
                    db.finish_job(jid, "running")  # release the claim so a later poll can retry
                    raise e
            elif time.monotonic() >= deadline:
                return {**db.get_job(jid), "assets": []}
            else:
                await asyncio.sleep(1)  # another poller is collecting
            continue
        if time.monotonic() >= deadline:
            return {**job, "assets": []}
        await asyncio.sleep(3)


def remove_background(asset_id: str) -> dict:
    """Sticker PNG (BRD: BiRefNet/RMBG). Runs rembg on the Mac CPU."""
    asset = db.get_asset(asset_id)
    if not asset:
        raise ValueError(f"Không có asset {asset_id}")
    src = config.STORAGE / asset["path"]
    dst = src.with_name(src.stem + "_sticker.png")
    dst.write_bytes(preprocess.cutout(src.read_bytes()))
    aid = db.add_asset(asset["project_id"], asset["job_id"], asset_id, "sticker_png", rel(dst), asset["seed"], {})
    return {"asset_id": aid, "path": dst}


# ---------- web-only flow: self-hosted LLM does what Claude does over MCP ----------

async def analyze_project(pid: str, category: str | None = None) -> None:
    from . import llm

    db.update_project(pid, status="analyzing", status_detail=None)
    try:
        submit_analysis(pid, await llm.analyze(pid, category))
    except Exception as e:  # surfaced in the UI; the user can retry or force a category
        db.update_project(pid, status="error", status_detail=str(e)[:2000])


async def edit_mindmap(pid: str, instruction_vi: str) -> int:
    from . import llm

    latest = db.latest_graph(pid)
    if not latest:
        raise ValueError("Project chưa có mind map")
    graph = validate_graph(await llm.revise_graph(latest[1], instruction_vi))
    return db.save_graph(pid, graph.model_dump(), source="llm")


async def auto_icons(pid: str) -> dict:
    latest = db.latest_graph(pid)
    # (re)write prompts when missing or too vague to draw ("startup-plan"): a picture needs a concrete object
    if latest and not any(len((n.get("icon_prompt_en") or "").split()) >= 3 for n in latest[1]["nodes"]):
        await edit_mindmap(pid, "Viết lại icon_prompt_en cho mọi khối: 4-8 từ tiếng Anh mô tả MỘT vật/cảnh cụ thể thể "
                                "hiện đúng ý nghĩa của khối đó, mỗi khối một hình khác nhau, không dùng dấu gạch nối, "
                                "không từ trừu tượng. Không đổi gì khác.")
    return await generate_icons(pid)


async def concept_to_character(pid: str) -> None:
    from . import llm

    latest = db.latest_graph(pid)
    if not latest:
        raise ValueError("Project chưa có mind map")
    db.save_profile(pid, (await llm.concept_profile(latest[1])).model_dump())
    db.update_project(pid, input_category="character_concept_mindmap")


async def variation_from_request(asset_id: str, request_vi: str, batch_size: int = 2,
                                 face_lock: float | None = None) -> dict:
    from . import llm

    asset = db.get_asset(asset_id)
    if not asset:
        raise ValueError(f"Không có asset {asset_id}")
    req = await llm.translate_request(request_vi, db.latest_profile(asset["project_id"]))
    return await generate_variation(asset_id, req.prompt_delta_en, req.strength, req.keep_pose, batch_size,
                                    face_lock=face_lock,
                                    colored_items=[(i.color_en, i.object_en) for i in req.colored_items])
