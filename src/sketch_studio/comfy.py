"""ComfyUI HTTP client + API-format workflows built only from core nodes (no custom nodes on Colab)."""

import random
import re
import uuid
from pathlib import Path

import httpx

from . import config

STYLE_PRESETS = {
    "anime": "anime style, cel shading, clean lineart, vibrant colors, 2d character illustration",
    "chibi": "chibi style, cute, big head, small body, 2d sticker art, thick clean outline",
    "flat_vector": "flat vector illustration, minimal shading, bold simple shapes, clean outlines",
    "cartoon": "western cartoon style, bold outlines, saturated colors, 2d animation character",
    "watercolor": "watercolor storybook illustration, soft edges, light paper texture",
}
BASE_SUFFIX = "full body character design, centered, simple plain white background, masterpiece, best quality"
BUST_SUFFIX = "close-up portrait, head and shoulders, expressive face, simple plain white background, masterpiece, best quality"

# Expression pack (BRD workflow 3, story_expression_pack): key -> (Vietnamese label, prompt)
# Eyes + mouth carry the emotion; explicit eyebrow words made SD paint thick brows over the bangs (T4 test).
EXPRESSIONS = {
    "happy": ("Vui vẻ", "happy, gentle smile, bright sparkling eyes"),
    "laugh": ("Cười lớn", "laughing out loud, wide open mouth, eyes closed with joy"),
    "sad": ("Buồn", "sad, downcast teary eyes, small frown, drooping mouth"),
    "cry": ("Khóc", "crying, streaming tears, sobbing, wavy mouth"),
    "angry": ("Tức giận", "angry, glaring eyes, pouting, clenched teeth, anger vein mark"),
    "surprised": ("Ngạc nhiên", "surprised, shocked, wide round eyes, small open mouth"),
    "scared": ("Sợ hãi", "scared, frightened, trembling, sweat drop, wavy mouth"),
    "shy": ("Ngượng ngùng", "embarrassed, blushing cheeks, shy smile, looking away"),
    "wink": ("Nháy mắt", "winking one eye, playful, tongue out"),
    "sleepy": ("Buồn ngủ", "sleepy, drowsy half-closed eyes, yawning"),
    "confident": ("Tự tin", "confident smirk, determined eyes"),
    "confused": ("Bối rối", "confused, puzzled, head tilt, question mark"),
}
EXPRESSION_NEGATIVE = "thick eyebrows, bushy eyebrows, unibrow, eyebrows over hair, extra eyebrows, mismatched eye color"
# Words that belong to a face's expression, not its identity: stripped from DNA traits when the prompt
# itself sets an expression or a change ("simple smile" in the DNA fought every sad/angry face).
_EXPRESSION_WORDS = re.compile(
    r"\b(?:and\s+)?(?:a\s+)?(?:simple\s+|big\s+|gentle\s+|happy\s+)?"
    r"(?:smile|smiling|smiles|grin|grinning|laughing|frown|frowning|crying|tears|angry|sad|cheerful)\b", re.I)
ICON_SUFFIX = "single 2d flat icon, sticker, thick outline, centered, plain white background, no text"
NEGATIVE = (
    "photo, realistic, 3d render, blurry, lowres, bad anatomy, extra limbs, deformed hands, "
    "text, watermark, signature, logo, emblem, flag, cropped, multiple characters, frame, border, "
    "scenery, detailed background, paint splash"
)

# fidelity -> (controlnet file, strength, end_percent)  [BRD: "Thanh trượt Fidelity"]
FIDELITY = {
    "high_fidelity": (config.CONTROLNET_LINEART, 1.0, 1.0),
    "balanced": (config.CONTROLNET_LINEART, 0.8, 0.75),
    "creative_fix": (config.CONTROLNET_SCRIBBLE, 0.6, 0.6),
}


# IP-Adapter (cubiq/ComfyUI_IPAdapter_plus) — locks a character's face from a reference crop.
FACE_MODEL = "ip-adapter-plus-face_sd15.safetensors"
FACE_PRESET = "PLUS FACE (portraits)"


class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, base_url: str):
        if not base_url:
            raise ComfyError("Chưa có COMFY_URL. Chạy notebook Colab rồi gọi tool set_comfy_url với link trycloudflare.")
        self.base = base_url.rstrip("/")

    async def _req(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.request(method, self.base + path, **kw)
        except httpx.HTTPError as e:
            raise ComfyError(f"Không kết nối được ComfyUI ({self.base}): {e!r}. Colab còn chạy không?") from e
        if r.status_code >= 400:
            raise ComfyError(f"ComfyUI {path} trả về {r.status_code}: {r.text[:1500]}")
        return r

    async def status(self) -> dict:
        stats = (await self._req("GET", "/system_stats")).json()
        ckpts = (await self._req("GET", "/models/checkpoints")).json()
        cns = (await self._req("GET", "/models/controlnet")).json()
        need = [config.CHECKPOINT, config.CONTROLNET_LINEART, config.CONTROLNET_SCRIBBLE]
        return {
            "devices": [d.get("name") for d in stats.get("devices", [])],
            "missing_models": [m for m in need if m not in ckpts + cns],
            "face_lock": await self.has_face_lock(),
        }

    async def has_face_lock(self) -> bool:
        """IP-Adapter custom node installed and its SD1.5 face model present (notebook installs both)."""
        info = (await self._req("GET", "/object_info/IPAdapterUnifiedLoader")).json()
        if "IPAdapterUnifiedLoader" not in info:
            return False
        r = await self._req("GET", "/models/ipadapter")
        return FACE_MODEL in r.json()

    async def upload(self, src: Path | bytes) -> str:
        data = src if isinstance(src, bytes) else src.read_bytes()
        name = f"studio_{uuid.uuid4().hex[:8]}.png"
        files = {"image": (name, data, "image/png")}
        r = await self._req("POST", "/upload/image", files=files, data={"overwrite": "true"})
        return r.json()["name"]

    async def queue(self, workflow: dict) -> str:
        r = await self._req("POST", "/prompt", json={"prompt": workflow, "client_id": "sketch-studio"})
        return r.json()["prompt_id"]

    async def history(self, prompt_id: str) -> dict | None:
        return (await self._req("GET", f"/history/{prompt_id}")).json().get(prompt_id)

    async def view(self, image: dict) -> bytes:
        params = {k: image[k] for k in ("filename", "subfolder", "type")}
        return (await self._req("GET", "/view", params=params)).content


def numbered(wf: dict) -> tuple[dict, dict]:
    """Rename readable node ids to "1","2",... like the ComfyUI frontend emits. Returns (workflow, {name: number})."""
    ids = {name: str(i) for i, name in enumerate(wf, 1)}
    out = {}
    for name, node in wf.items():
        inputs = {k: [ids[v[0]], v[1]] if isinstance(v, list) and len(v) == 2 and v[0] in ids else v
                  for k, v in node["inputs"].items()}
        out[ids[name]] = {**node, "inputs": inputs}
    return out, ids


def seed_or_random(seed: int | None) -> int:
    return seed if seed is not None else random.randint(0, 2**32 - 1)


def _encode(wf: dict, name: str, text: str) -> None:
    """CLIPTextEncode; ' BREAK ' splits the text into chunks encoded separately and joined with
    ConditioningConcat (like A1111's BREAK), so a colour stays on its own object instead of bleeding."""
    chunks = [c.strip() for c in text.split(" BREAK ") if c.strip()] or [""]
    if len(chunks) == 1:
        wf[name] = {"class_type": "CLIPTextEncode", "inputs": {"text": chunks[0], "clip": ["ckpt", 1]}}
        return
    for i, chunk in enumerate(chunks):
        wf[f"{name}_{i}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": chunk, "clip": ["ckpt", 1]}}
    prev = [f"{name}_0", 0]
    for i in range(1, len(chunks)):
        key = name if i == len(chunks) - 1 else f"{name}_cat{i}"
        wf[key] = {"class_type": "ConditioningConcat",
                   "inputs": {"conditioning_to": prev, "conditioning_from": [f"{name}_{i}", 0]}}
        prev = [key, 0]


def _base(ckpt: str, positive: str, negative: str) -> dict:
    wf = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    _encode(wf, "pos", positive)
    _encode(wf, "neg", negative)
    return wf


_CLOTHING = re.compile(r"\b(dress|shirt|skirt|outfit|clothes|jacket|coat|hoodie|uniform|suit|pants|shorts)\b", re.I)


def bind_colors(delta: str, items: list[tuple[str, str]]) -> tuple[str, str, str]:
    """(red, balloon), (pink, bow) -> the delta unchanged (stays first in the prompt), one extra BREAK chunk per
    coloured object to reinforce it, and negatives against cross-bleed ("pink balloon", "red bow", "pink dress").
    T4 test: stripping the colours from the delta made the balloon vanish; the negatives stopped the pink dress."""
    plain = delta
    chunks = " BREAK ".join(f"({c} {o}:1.35)" for c, o in items)
    colors = {c.lower() for c, _ in items}
    neg = [f"{other} {o}" for c, o in items for other in colors if other != c.lower()]
    if not any(_CLOTHING.search(o) for _, o in items):
        neg += [f"{c} dress, {c} clothes" for c in colors]
    return plain, chunks, ", ".join(neg)


def _sampler(latent: list, pos: list, neg: list, seed: int, steps: int, denoise: float, cfg: float = 6.5,
             model: list = ("ckpt", 0)) -> dict:
    return {
        "class_type": "KSampler",
        "inputs": {
            "model": list(model), "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "positive": pos, "negative": neg, "latent_image": latent, "denoise": denoise,
        },
    }


def _controlnet(wf: dict, hint_image: str, fidelity: str) -> tuple[list, list]:
    cn_file, strength, end = FIDELITY[fidelity]
    wf["hint"] = {"class_type": "LoadImage", "inputs": {"image": hint_image}}
    wf["cn"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn_file}}
    wf["cn_apply"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "control_net": ["cn", 0], "image": ["hint", 0],
            "strength": strength, "start_percent": 0.0, "end_percent": end,
        },
    }
    return ["cn_apply", 0], ["cn_apply", 1]


def _face_lock(wf: dict, face_image: str | None, weight: float, end_at: float) -> list:
    """Patch the model with IP-Adapter PLUS FACE from a face crop; returns the model link to sample with."""
    if not face_image or weight <= 0:
        return ["ckpt", 0]
    wf["face_ref"] = {"class_type": "LoadImage", "inputs": {"image": face_image}}
    wf["ipa_loader"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["ckpt", 0], "preset": FACE_PRESET}}
    wf["ipa"] = {
        "class_type": "IPAdapterAdvanced",
        "inputs": {
            "model": ["ipa_loader", 0], "ipadapter": ["ipa_loader", 1], "image": ["face_ref", 0],
            "weight": weight, "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": end_at, "embeds_scaling": "V only",
        },
    }
    return ["ipa", 0]


def character_prompt(profile: dict, style: str, extra: str = "", suffix: str = BASE_SUFFIX,
                     focus: str = "", focus_weight: float = 1.3) -> str:
    """focus = the change being asked for (emotion, "holding a red balloon"): it goes FIRST and up-weighted.
    At the end of a long prompt SD1.5 let it lose to the DNA colours (red balloon came out pink)."""
    p = profile["color_palette"]
    traits, archetype = profile["immutable_traits_en"], profile["character_archetype"]
    if focus:
        def neutral(t):
            return _EXPRESSION_WORDS.sub("", t).strip(" ,")
        traits = [t for t in map(neutral, traits) if t and t.lower() not in ("face", "eyes", "mouth")]
        archetype = neutral(archetype)
    parts = [
        f"({focus}:{focus_weight})" if focus else "",
        STYLE_PRESETS[style],
        archetype,
        *traits,
        profile["mutable_outfit_en"],
        # eye/hair colour weighted a little: strong emotion prompts drifted black eyes to blue
        f"({p['hair_color']} hair, {p['eye_color']} eyes:1.15), {p['primary_outfit_color']} outfit, {p['accent_color']} accents",
        extra,
        suffix,
    ]
    return ", ".join(x.strip() for x in parts if x and x.strip())


def character_workflow(
    positive: str, size: tuple[int, int], batch: int, seed: int,
    hint_image: str | None, fidelity: str, hires: bool = True,
) -> dict:
    """Workflow 1 (sketch_to_2d_v1): ControlNet lineart/scribble + optional hires pass.
    hint_image=None -> plain txt2img (concept mind map -> character)."""
    wf = _base(config.CHECKPOINT, positive, NEGATIVE)
    pos, neg = (["pos", 0], ["neg", 0]) if hint_image is None else _controlnet(wf, hint_image, fidelity)
    wf["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": batch}}
    wf["sample"] = _sampler(["latent", 0], pos, neg, seed, 24, 1.0)
    out = ["sample", 0]
    if hires:
        wf["up"] = {"class_type": "LatentUpscaleBy", "inputs": {"samples": out, "upscale_method": "nearest-exact", "scale_by": 1.5}}
        wf["sample2"] = _sampler(["up", 0], pos, neg, seed, 14, 0.45)
        out = ["sample2", 0]
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": out, "vae": ["ckpt", 2]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "studio/char"}}
    return wf


def variation_workflow(
    positive: str, source_image: str, strength: float, batch: int, seed: int,
    hint_image: str | None = None, fidelity: str = "balanced", face_image: str | None = None, face_weight: float = 0.0,
    negative_extra: str = "",
) -> dict:
    """Workflow 2-lite (character_variation): img2img from an approved asset keeps identity/colors;
    optional ControlNet on the original sketch keeps the pose; IP-Adapter optionally locks the face."""
    wf = _base(config.CHECKPOINT, positive, f"{NEGATIVE}, {negative_extra}" if negative_extra else NEGATIVE)
    pos, neg = (["pos", 0], ["neg", 0]) if hint_image is None else _controlnet(wf, hint_image, fidelity)
    wf["src"] = {"class_type": "LoadImage", "inputs": {"image": source_image}}
    wf["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["src", 0], "vae": ["ckpt", 2]}}
    wf["rep"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["enc", 0], "amount": batch}}
    model = _face_lock(wf, face_image, face_weight, end_at=1.0)
    wf["sample"] = _sampler(["rep", 0], pos, neg, seed, 28, strength, model=model)
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "studio/var"}}
    return wf


def icons_workflow(prompts: dict[str, str], seed: int) -> dict:
    """Workflow 6 (mindmap_illustrated_sketchnote): one 512x512 icon branch per mind-map node.
    SaveImage node ids are 'save_<node_id>' so outputs map back to nodes."""
    wf = _base(config.CHECKPOINT, "", NEGATIVE)
    del wf["pos"]
    wf["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}}
    for i, (node_id, prompt) in enumerate(prompts.items()):
        wf[f"pos_{node_id}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": f"{prompt}, {ICON_SUFFIX}", "clip": ["ckpt", 1]}}
        wf[f"sample_{node_id}"] = _sampler(["latent", 0], [f"pos_{node_id}", 0], ["neg", 0], seed + i, 20, 1.0, cfg=7.0)
        wf[f"decode_{node_id}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"sample_{node_id}", 0], "vae": ["ckpt", 2]}}
        wf[f"save_{node_id}"] = {"class_type": "SaveImage", "inputs": {"images": [f"decode_{node_id}", 0], "filename_prefix": "studio/icon"}}
    return wf


def expressions_workflow(prompts: dict[str, str], source_image: str, denoise: float, seed: int,
                         face_image: str | None = None, face_weight: float = 0.0) -> dict:
    """Expression pack: one img2img branch per emotion from the same source latent and the SAME seed,
    so only the prompt differs -> consistent identity. SaveImage ids are 'save_<emotion key>'."""
    wf = _base(config.CHECKPOINT, "", f"{NEGATIVE}, {EXPRESSION_NEGATIVE}")
    del wf["pos"]
    wf["src"] = {"class_type": "LoadImage", "inputs": {"image": source_image}}
    wf["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["src", 0], "vae": ["ckpt", 2]}}
    # end_at < 1: the adapter fixes identity early, the later steps follow the emotion prompt (PLUS FACE
    # would otherwise also copy the reference's expression). 0.8 kept faces right but emotions too faint (T4 test).
    model = _face_lock(wf, face_image, face_weight, end_at=0.6)
    for key, prompt in prompts.items():
        wf[f"pos_{key}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}}
        wf[f"sample_{key}"] = _sampler(["enc", 0], [f"pos_{key}", 0], ["neg", 0], seed, 28, denoise, model=model)
        wf[f"decode_{key}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"sample_{key}", 0], "vae": ["ckpt", 2]}}
        wf[f"save_{key}"] = {"class_type": "SaveImage", "inputs": {"images": [f"decode_{key}", 0], "filename_prefix": "studio/expr"}}
    return wf
