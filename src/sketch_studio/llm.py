"""Self-hosted VLM (Qwen via Ollama on Colab) for the web-only flow — does the jobs Claude does over MCP:
classify, read the mind map, write Character DNA, revise graphs and translate requests into prompts."""

import asyncio
import base64
import copy
import json
import math
import re
import unicodedata

import httpx
from pydantic import BaseModel, Field, ValidationError

from . import config, db, preprocess
from .schemas import (
    CharacterDNAProfile, InputCategory, MindMapGraphStructure, UniversalSketchAnalysisResponse,
)


class LLMError(RuntimeError):
    pass


_locks: dict = {}


def _lock() -> asyncio.Lock:
    """One LLM call at a time per event loop (the web server has one; tests create several)."""
    loop = asyncio.get_running_loop()
    return _locks.setdefault(loop, asyncio.Lock())


def llm_schema(model: type[BaseModel], drop: set[str] = frozenset()) -> dict:
    """JSON schema with $refs inlined and server-managed fields removed (smaller grammar for small models)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            out = {k: walk(v) for k, v in node.items() if k != "title"}
            if "properties" in out:
                out["properties"] = {k: v for k, v in out["properties"].items() if k not in drop}
                out["required"] = [k for k in out.get("required", []) if k not in drop]
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


class Ollama:
    def __init__(self, url: str, model: str = ""):
        if not url:
            raise LLMError("Chưa có LLM_URL. Chạy cell Ollama trong notebook Colab rồi dán link vào Cài đặt.")
        self.base, self.model = url.rstrip("/"), model

    async def _chat(self, body: dict) -> str:
        """Streamed /api/chat. trycloudflare cuts any response that sends nothing for ~100 s (HTTP 524) and a 9B
        model on a shared T4 often needs longer, so tokens are streamed. Calls are serialized here so a request
        never sits silently in Ollama's queue behind another one."""
        async with _lock():
            parts = []
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30)) as c:
                    async with c.stream("POST", self.base + "/api/chat", json={**body, "stream": True}) as r:
                        if r.status_code >= 400:
                            raise LLMError(f"LLM /api/chat trả về {r.status_code}: {(await r.aread())[:800]!r}")
                        async for line in r.aiter_lines():
                            if not line.strip():
                                continue
                            chunk = json.loads(line)
                            if "error" in chunk:
                                raise LLMError(f"Ollama báo lỗi: {chunk['error']}")
                            parts.append(chunk.get("message", {}).get("content", ""))
            except httpx.HTTPError as e:
                raise LLMError(f"Không kết nối được LLM ({self.base}): {e!r}. Colab còn chạy không?") from e
            return "".join(parts)

    async def models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(self.base + "/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"Không kết nối được LLM ({self.base}): {e!r}. Colab còn chạy không?") from e

    async def resolve_model(self) -> str:
        if not self.model:
            names = await self.models()
            if not names:
                raise LLMError("Ollama chưa có model nào — chạy lại cell tải model trong notebook")
            self.model = names[0]
        return self.model

    async def chat_json(self, system: str, user: str, out: type[BaseModel], images: list[bytes] = (),
                        drop: set[str] = frozenset(), temperature: float = 0.2) -> BaseModel:
        """One structured call; on invalid JSON retry once with the validation error appended."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user, "images": [base64.b64encode(i).decode() for i in images]},
        ]
        body = {
            "model": await self.resolve_model(), "messages": messages, "format": llm_schema(out, drop),
            "think": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_ctx": 16384},
        }
        for attempt in range(2):
            content = await self._chat(body)
            try:
                return out.model_validate_json(content)
            except ValidationError as e:
                if attempt:
                    raise LLMError(f"LLM trả JSON không hợp lệ: {e.errors()[:3]}\n{content[:500]}") from e
                messages += [{"role": "assistant", "content": content},
                             {"role": "user", "content": f"JSON chưa đúng schema: {e.errors()[:3]}. Trả lại JSON đã sửa."}]


def client() -> Ollama:
    return Ollama(db.get_setting("llm_url", config.LLM_URL), db.get_setting("llm_model", config.LLM_MODEL))


# ---------------- prompts ----------------

class Classification(BaseModel):
    detected_input_category: InputCategory
    reason_vi: str = Field(description="Một câu giải thích ngắn")


class ColoredItem(BaseModel):
    color_en: str = Field(description="English colour word, e.g. red")
    object_en: str = Field(description="the object that has this colour, e.g. balloon")


class VariationRequest(BaseModel):
    prompt_delta_en: str = Field(description="Short English Stable Diffusion keywords describing ONLY the change")
    colored_items: list[ColoredItem] = Field(description="every object the user gave a colour; empty if none")
    strength: float = Field(ge=0.3, le=0.8, description="0.35 subtle, 0.55 normal, 0.75 big change")
    keep_pose: bool


CLASSIFY = """You classify a photo of a hand-drawn paper sketch into exactly one category:
- character_or_object_sketch: a drawing of a character, creature or object (figure, face, body, item).
- mindmap_or_flowchart_diagram: shapes (circles, triangles, boxes, diamonds) containing handwritten text, connected by lines/arrows.
- character_concept_mindmap: a mind map whose branches describe a character's looks/outfit/weapon/personality.
Answer in JSON."""

MINDMAP = """You digitize a photo of a hand-drawn mind map / flowchart into a graph.
Rules:
- One node per drawn shape. shape_type = the shape actually drawn (circle, triangle, rectangle, diamond, cloud, pill).
- text_content_vi: copy the handwritten text EXACTLY, keep Vietnamese diacritics; do not translate or summarize.
- x_norm, y_norm: center of the shape on the page, 0-1000 (0,0 = top-left of the image).
- Edges follow the drawn lines. arrow_type: directed (one arrowhead), bidirectional, undirected_line (no head), dashed.
  For directed edges source = tail, target = arrow head. relation_label_vi = text written on the line, else null.
- suggested_color_hex: pleasant pastel color per branch/topic (e.g. #FFADAD), same color for siblings.
- icon_prompt_en: 4-8 English words describing ONE concrete object or scene that shows THIS node's meaning,
  to draw as a flat icon. Format example only (do not reuse): node "Tiết kiệm" -> "piggy bank with falling coins".
  Plain words, no hyphens, no abstract terms like "plan" or "strategy". Every node gets a different picture.
- title_vi: short Vietnamese title for the whole map.
- ai_expansion_suggestions_vi: 3-5 concrete Vietnamese suggestions to extend/improve the map.
- node_id: n1, n2, ...; edge_id: e1, e2, ..."""

CHARACTER = """You analyze a hand-drawn character sketch and write its Character DNA for Stable Diffusion 1.5 (English keywords).
Describe the FINISHED character the sketch is meant to be, not the drawing itself: never use words like stick,
stick figure, line, circle, trapezoid, sketch, doodle, simple drawing. A round head with dots is "a cute girl with
big eyes", a trapezoid body is "a dress", a small circle held in the hand is "a balloon" or "a ball".
- character_archetype: short English phrase (e.g. "cheerful little girl", "young space explorer").
- immutable_traits_en: 3-6 short visual traits that define identity (face shape, eyes, hair, headwear, body).
  No facial expression (smile, frown...) and no hand-held items here: those change between images.
- mutable_outfit_en: one English phrase describing clothing and any held items.
- color_palette: choose fitting colors as plain English color words (e.g. "dark brown"), never hex codes.
- sketch_complexity_score 1-10; recommended_fidelity_mode: high_fidelity for clean detailed lines,
  balanced for normal sketches, creative_fix for rough/messy scribbles.
- contextual_suggestions: 4-6 chips. label_vi in Vietnamese, prompt_delta_en English keywords for the change only.
- character_id: snake_case slug."""

_COLORS = {
    "black": (0, 0, 0), "dark gray": (64, 64, 64), "gray": (128, 128, 128), "silver": (192, 192, 192),
    "white": (255, 255, 255), "red": (220, 40, 40), "dark red": (128, 20, 20), "pink": (255, 160, 190),
    "orange": (255, 140, 0), "yellow": (250, 220, 40), "gold": (212, 175, 55), "beige": (225, 205, 170),
    "brown": (140, 90, 50), "dark brown": (80, 50, 30), "green": (60, 170, 70), "dark green": (20, 90, 40),
    "teal": (0, 128, 128), "light blue": (140, 200, 240), "blue": (40, 90, 220), "navy": (20, 30, 90),
    "purple": (130, 60, 180), "lavender": (200, 170, 230),
}


def color_word(value: str) -> str:
    """'#8B7355' -> 'brown'. SD1.5 prompts need words; small models sometimes answer with hex codes."""
    v = value.strip().lstrip("#")
    if len(v) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in v):
        return value
    rgb = tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
    return min(_COLORS, key=lambda n: sum((a - b) ** 2 for a, b in zip(_COLORS[n], rgb)))


def _clean_profile(p: CharacterDNAProfile) -> CharacterDNAProfile:
    for field in ("hair_color", "eye_color", "primary_outfit_color", "accent_color"):
        setattr(p.color_palette, field, color_word(getattr(p.color_palette, field)))
    return p


CONCEPT = """The user described a character as a mind map (JSON below). Merge every branch into one Character DNA for
Stable Diffusion 1.5 (English keywords). Every branch detail must appear in traits, outfit or palette.
Choose recommended_fidelity_mode = creative_fix and sketch_complexity_score = 3.
contextual_suggestions: 4-6 chips, label_vi Vietnamese, prompt_delta_en English."""

REVISE = """You edit a mind map graph (JSON). Apply the user's Vietnamese instruction and return the FULL updated graph.
Keep existing node_id/edge_id, texts and relation_label_vi unless the instruction changes them. New nodes: next
free ids (any x_norm/y_norm, the server lays them out). Keep Vietnamese diacritics.
Refresh ai_expansion_suggestions_vi for the new state."""

TRANSLATE = """Turn the user's Vietnamese request about changing a 2D character image into Stable Diffusion 1.5 keywords.
Only describe the change (expression, pose, outfit, accessory, background), not the whole character.
keep_pose=false when the request changes the pose/action.
colored_items: one entry per object the user gives a colour ("bóng bay màu đỏ" -> red / balloon)."""

DROP_GRAPH = {"icon_url", "mermaid_code"}


def _image(pid: str) -> bytes:
    return preprocess.thumbnail_jpeg(config.STORAGE / db.get_project(pid)["sketch_raw"], 1280, 90)


async def analyze(pid: str, category: str | None = None) -> UniversalSketchAnalysisResponse:
    c, img = client(), _image(pid)
    if not category:
        category = (await c.chat_json(CLASSIFY, "Classify this sketch.", Classification, [img])).detected_input_category
    graph = profile = None
    if category != "character_or_object_sketch":
        graph = await c.chat_json(MINDMAP, "Digitize this mind map.", MindMapGraphStructure, [img], DROP_GRAPH)
        await proofread(c, img, graph)
    if category == "character_or_object_sketch":
        profile = _clean_profile(await c.chat_json(CHARACTER, "Analyze this character sketch.", CharacterDNAProfile, [img]))
    elif category == "character_concept_mindmap":
        profile = _clean_profile(
            await c.chat_json(CONCEPT, graph.model_dump_json(include={"nodes", "edges"}), CharacterDNAProfile))
    return UniversalSketchAnalysisResponse(detected_input_category=category, mindmap_graph=graph, character_profile=profile)


class TextFix(BaseModel):
    id: str
    text_vi: str


class TextFixes(BaseModel):
    items: list[TextFix]


PROOFREAD = """You check Vietnamese text that was read from the handwritten mind map in the image.
For each item compare with the image and with correct Vietnamese spelling, and fix ONLY wrong or missing
diacritics (dấu: sắc, huyền, hỏi, ngã, nặng, mũ, móc) and d/đ. Do not change words, wording or line breaks.
Example: "Sản phẩm dầu tiên" -> "Sản phẩm đầu tiên". Return every item with the same id."""


def _base_letters(text: str) -> str:
    """'Sản phẩm đầu' -> 'san pham dau': what must stay equal for a fix to count as diacritics-only."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return " ".join("".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower().split())


def _relayout(old: str, new: str) -> str:
    """Put the corrected words back into the original line breaks ("Sản phẩm\\ndầu tiên" -> "Sản phẩm\\nđầu tiên")."""
    seps = re.findall(r"\s+", old.strip())
    words = new.split()
    if len(words) != len(seps) + 1:
        return new
    return "".join(w + (seps[i] if i < len(seps) else "") for i, w in enumerate(words))


async def proofread(c: "Ollama", img: bytes, graph: MindMapGraphStructure) -> None:
    """Second look at the texts: a 9B model drops/garbles Vietnamese diacritics on some runs ("dầu tiên").
    Only diacritic changes are accepted, so a proofreading pass can never rewrite what the user wrote."""
    targets = {n.node_id: n for n in graph.nodes}
    targets.update({e.edge_id: e for e in graph.edges if e.relation_label_vi})
    texts = {i: (t.text_content_vi if hasattr(t, "text_content_vi") else t.relation_label_vi) for i, t in targets.items()}
    # one line per item, no JSON escapes: with "Sản phẩm\\ndầu tiên" the model echoed the text unchanged (T4 test)
    listing = "\n".join(f"{i}: {' '.join(t.split())}" for i, t in texts.items())
    try:
        fixes = await c.chat_json(PROOFREAD, listing, TextFixes, [img], temperature=0.0)
    except LLMError:
        return  # proofreading is best-effort; the first reading stands
    for fix in fixes.items:
        old = texts.get(fix.id)
        if old is None or _base_letters(fix.text_vi) != _base_letters(old):
            continue
        fixed = _relayout(old, fix.text_vi)
        if fixed == old:
            continue
        target = targets[fix.id]
        if hasattr(target, "text_content_vi"):
            target.text_content_vi = fixed
        else:
            target.relation_label_vi = fixed


async def revise_graph(graph: dict, instruction_vi: str) -> MindMapGraphStructure:
    payload = {k: v for k, v in graph.items() if k != "mermaid_code"}
    user = f"Instruction: {instruction_vi}\n\nGraph JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    new = await client().chat_json(REVISE, user, MindMapGraphStructure, drop=DROP_GRAPH)
    return merge_revision(graph, new)


def _free_spot(px: int, py: int, taken: list[tuple[int, int]]) -> tuple[int, int]:
    """First spot around (px, py), preferring below the parent, whose box doesn't hit an existing node."""
    def clear(x, y):
        return all(abs(x - tx) >= 125 or abs(y - ty) >= 95 for tx, ty in taken)

    for r in (170, 250, 330):
        for deg in (90, 55, 125, 20, 160, -20, 200, -55, 235):
            x = round(px + r * math.cos(math.radians(deg)))
            y = round(py + r * math.sin(math.radians(deg)))
            if 30 <= x <= 970 and 30 <= y <= 970 and clear(x, y):
                return x, y
    return min(max(px, 30), 970), min(py + 170, 970)


def merge_revision(old: dict, new: MindMapGraphStructure) -> MindMapGraphStructure:
    """Small models rewrite what they were asked to keep. Restore what the server owns: positions and icons
    of surviving nodes, labels of surviving edges; lay out new nodes around their parent without overlaps."""
    old_nodes = {n["node_id"]: n for n in old["nodes"]}
    old_labels = {(e["source_node_id"], e["target_node_id"]): e.get("relation_label_vi") for e in old["edges"]}
    pos = {}
    for n in new.nodes:
        if n.node_id in old_nodes:
            o = old_nodes[n.node_id]
            n.x_norm, n.y_norm, n.icon_url = o["x_norm"], o["y_norm"], o.get("icon_url")
            pos[n.node_id] = (n.x_norm, n.y_norm)
        else:
            n.icon_url = None
    for e in new.edges:
        if not e.relation_label_vi:
            e.relation_label_vi = old_labels.get((e.source_node_id, e.target_node_id))
    parent = {e.target_node_id: e.source_node_id for e in new.edges}
    for n in new.nodes:
        if n.node_id not in old_nodes:
            px, py = pos.get(parent.get(n.node_id), (500, 500))
            n.x_norm, n.y_norm = _free_spot(px, py, list(pos.values()))
            pos[n.node_id] = (n.x_norm, n.y_norm)
    return new


async def concept_profile(graph: dict) -> CharacterDNAProfile:
    payload = json.dumps({k: graph[k] for k in ("nodes", "edges")}, ensure_ascii=False)
    return _clean_profile(await client().chat_json(CONCEPT, payload, CharacterDNAProfile))


async def translate_request(request_vi: str, profile: dict | None) -> VariationRequest:
    ctx = f"Character: {profile['character_archetype']}, traits: {profile['immutable_traits_en']}\n" if profile else ""
    return await client().chat_json(TRANSLATE, f"{ctx}Request: {request_vi}", VariationRequest)
