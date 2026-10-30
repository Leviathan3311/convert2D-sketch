"""Minimal ComfyUI stand-in. Validates API-format workflows against the core node signatures
(copied from ComfyUI nodes.py) and returns fake PNGs, so the full pipeline runs without a GPU."""

import io
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

# class -> (required inputs, number of outputs)
NODES = {
    "CheckpointLoaderSimple": ({"ckpt_name"}, 3),
    "CLIPTextEncode": ({"text", "clip"}, 1),
    "EmptyLatentImage": ({"width", "height", "batch_size"}, 1),
    "KSampler": ({"model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image", "denoise"}, 1),
    "VAEDecode": ({"samples", "vae"}, 1),
    "VAEEncode": ({"pixels", "vae"}, 1),
    "SaveImage": ({"images", "filename_prefix"}, 1),
    "LoadImage": ({"image"}, 2),
    "ControlNetLoader": ({"control_net_name"}, 1),
    "ControlNetApplyAdvanced": ({"positive", "negative", "control_net", "image", "strength", "start_percent", "end_percent"}, 2),
    "LatentUpscaleBy": ({"samples", "upscale_method", "scale_by"}, 1),
    "RepeatLatentBatch": ({"samples", "amount"}, 1),
    "ConditioningConcat": ({"conditioning_to", "conditioning_from"}, 1),
    # cubiq/ComfyUI_IPAdapter_plus (IPAdapterPlus.py)
    "IPAdapterUnifiedLoader": ({"model", "preset"}, 2),
    "IPAdapterAdvanced": ({"model", "ipadapter", "image", "weight", "weight_type", "combine_embeds",
                           "start_at", "end_at", "embeds_scaling"}, 1),
}
IPA_PRESETS = ["LIGHT - SD1.5 only (low strength)", "STANDARD (medium strength)", "VIT-G (medium strength)",
               "PLUS (high strength)", "PLUS FACE (portraits)", "FULL FACE - SD1.5 only (portraits stronger)"]
MODELS = {
    "checkpoints": ["DreamShaper_8_pruned.safetensors"],
    "controlnet": ["control_v11p_sd15_lineart_fp16.safetensors", "control_v11p_sd15_scribble_fp16.safetensors"],
    "ipadapter": ["ip-adapter-plus-face_sd15.safetensors"],
}

app = FastAPI()
app.state.uploads = {}
app.state.history = {}
app.state.prompts = []
app.state.ipadapter = True  # tests flip this to simulate a Colab without the custom node


def validate(wf: dict) -> list[str]:
    errs = []
    for nid, node in wf.items():
        if not nid.isdigit():
            errs.append(f"{nid}: non-numeric id")
        spec = NODES.get(node.get("class_type"))
        if not spec:
            errs.append(f"{nid}: unknown class {node.get('class_type')}")
            continue
        required, _ = spec
        missing = required - set(node["inputs"])
        if missing:
            errs.append(f"{nid}: missing {missing}")
        for k, v in node["inputs"].items():
            if isinstance(v, list):
                src = wf.get(v[0])
                if src is None or v[1] >= NODES[src["class_type"]][1]:
                    errs.append(f"{nid}.{k}: bad link {v}")
        ins = node["inputs"]
        if node["class_type"] == "LoadImage" and ins["image"] not in app.state.uploads:
            errs.append(f"{nid}: image {ins['image']} not uploaded")
        if node["class_type"] == "CheckpointLoaderSimple" and ins["ckpt_name"] not in MODELS["checkpoints"]:
            errs.append(f"{nid}: ckpt missing")
        if node["class_type"] == "ControlNetLoader" and ins["control_net_name"] not in MODELS["controlnet"]:
            errs.append(f"{nid}: controlnet missing")
        if node["class_type"] == "IPAdapterUnifiedLoader" and ins["preset"] not in IPA_PRESETS:
            errs.append(f"{nid}: bad preset")
        if node["class_type"].startswith("IPAdapter") and not app.state.ipadapter:
            errs.append(f"{nid}: IPAdapter node not installed")
        if node["class_type"] == "KSampler" and not 0 <= ins["denoise"] <= 1:
            errs.append(f"{nid}: denoise out of range")
    return errs


def batch_of(wf: dict, save_id: str) -> int:
    """Follow SaveImage -> VAEDecode -> ... back to the latent to know how many images to emit."""
    nid = save_id
    while True:
        node = wf[nid]
        ins = node["inputs"]
        if node["class_type"] == "EmptyLatentImage":
            return ins["batch_size"]
        if node["class_type"] == "RepeatLatentBatch":
            return ins["amount"]
        if node["class_type"] == "VAEEncode":
            return 1
        nid = (ins.get("images") or ins.get("samples") or ins.get("latent_image"))[0]


@app.get("/system_stats")
def stats():
    return {"system": {}, "devices": [{"name": "mock T4"}]}


@app.get("/object_info/{cls}")
def object_info(cls: str):
    if cls.startswith("IPAdapter") and not app.state.ipadapter:
        return {}
    return {cls: {"input": {"required": {k: [] for k in NODES[cls][0]}}}} if cls in NODES else {}


@app.get("/models/{folder}")
def models(folder: str):
    return MODELS.get(folder, [])


@app.post("/upload/image")
async def upload(image: UploadFile = File(...), overwrite: str = Form("false")):
    app.state.uploads[image.filename] = await image.read()
    return {"name": image.filename, "subfolder": "", "type": "input"}


@app.post("/prompt")
async def prompt(req: Request):
    wf = (await req.json())["prompt"]
    errs = validate(wf)
    if errs:
        return JSONResponse({"error": {"type": "prompt_outputs_failed_validation"}, "node_errors": errs}, status_code=400)
    pid = str(uuid.uuid4())
    app.state.prompts.append(wf)
    outputs = {}
    for nid, node in wf.items():
        if node["class_type"] == "SaveImage":
            outputs[nid] = {"images": [{"filename": f"{nid}_{i}.png", "subfolder": "studio", "type": "output"} for i in range(batch_of(wf, nid))]}
    # Not in history on first poll -> exercises the "still running" path.
    app.state.history[pid] = {"polls": 0, "entry": {"outputs": outputs, "status": {
        "status_str": "success", "completed": True,
        "messages": [["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 9000}]]}}}
    return {"prompt_id": pid, "number": 1, "node_errors": {}}


@app.get("/history/{pid}")
def history(pid: str):
    h = app.state.history.get(pid)
    if not h:
        return {}
    h["polls"] += 1
    return {pid: h["entry"]} if h["polls"] > 1 else {}


@app.get("/view")
def view(filename: str, subfolder: str = "", type: str = "output"):
    buf = io.BytesIO()
    Image.new("RGB", (64, 96), (200, 120, 90)).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ---------------- Ollama stand-in (same app, so one URL serves both in tests) ----------------
import json  # noqa: E402

app.state.chat_calls = []
app.state.category = "mindmap_or_flowchart_diagram"
app.state.fail_once = False


@app.get("/api/tags")
def tags():
    return {"models": [{"name": "qwen3.5:9b-q4_K_M"}]}


@app.post("/api/chat")
async def chat(req: Request):
    from test_pipeline import GRAPH, PROFILE

    body = await req.json()
    app.state.chat_calls.append(body)
    assert body["think"] is False and body["stream"] is True and "$ref" not in json.dumps(body["format"])
    props = body["format"]["properties"]
    last = body["messages"][-1]["content"]
    if app.state.fail_once:
        app.state.fail_once = False
        content = '{"oops": 1}'
    elif "detected_input_category" in props:
        content = json.dumps({"detected_input_category": app.state.category, "reason_vi": "có hình khối"})
    elif "items" in props:  # proofreading pass: fix one diacritic, try (and fail) to rewrite another node
        items = []
        for line in last.splitlines():
            node_id, text = line.split(": ", 1)
            text = text.replace("Quyêt đinh", "Quyết định")
            items.append({"id": node_id, "text_vi": "Khách hàng" if node_id == "n2" else text})
        content = json.dumps({"items": items}, ensure_ascii=False)
    elif "nodes" in props:
        graph = json.loads(json.dumps(GRAPH))
        if not last.startswith("Instruction:"):
            graph["nodes"][2]["text_content_vi"] = "Quyêt đinh"  # what a 9B model sometimes reads
        if last.startswith("Instruction:"):
            graph["nodes"].append({**graph["nodes"][2], "node_id": "n4", "text_content_vi": "Ngân sách"})
            graph["edges"].append({"edge_id": "e3", "source_node_id": "n1", "target_node_id": "n4", "arrow_type": "directed"})
        content = json.dumps(graph, ensure_ascii=False)
    elif "character_archetype" in props:
        content = json.dumps(PROFILE)
    else:
        colored = [{"color_en": "red", "object_en": "balloon"}] if "bóng bay" in last else []
        content = json.dumps({"prompt_delta_en": "laughing, wool hat" + (", holding a red balloon" if colored else ""),
                              "strength": 0.6, "keep_pose": True, "colored_items": colored})
    return {"model": body["model"], "message": {"role": "assistant", "content": content}, "done": True}
