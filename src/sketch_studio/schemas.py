"""Pydantic schemas from BRD v2.2 §4 (VLM router envelope for both engines)."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --- 1. Mind map / block diagram branch ---
class MindMapNode(BaseModel):
    node_id: str = Field(description="ID duy nhất của khối, ví dụ: node_1")
    shape_type: Literal["triangle", "circle", "rectangle", "diamond", "cloud", "pill"] = Field(
        description="Hình dạng khối mà người dùng đã vẽ trên giấy"
    )
    text_content_vi: str = Field(description="Nội dung chữ viết bên trong hoặc cạnh khối (giữ nguyên tiếng Việt)")
    x_norm: int = Field(ge=0, le=1000, description="Tọa độ tâm X chuẩn hóa (0-1000) theo bố cục giấy gốc")
    y_norm: int = Field(ge=0, le=1000, description="Tọa độ tâm Y chuẩn hóa (0-1000) theo bố cục giấy gốc")
    suggested_color_hex: str = Field(description="Mã màu 2D hài hòa gợi ý cho khối này, ví dụ: #FFADAD")
    icon_prompt_en: Optional[str] = Field(
        default=None,
        description="Prompt tiếng Anh ngắn để ComfyUI vẽ icon 2D minh họa cho nút này (nếu bật chế độ Illustrated Mind Map)",
    )
    icon_url: Optional[str] = Field(default=None, description="Do server điền sau khi sinh icon; để trống")


class MindMapEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_label_vi: Optional[str] = Field(default=None, description="Chữ ghi trên mũi tên nối (nếu có)")
    arrow_type: Literal["directed", "bidirectional", "undirected_line", "dashed"]


class MindMapGraphStructure(BaseModel):
    title_vi: str
    nodes: List[MindMapNode]
    edges: List[MindMapEdge]
    mermaid_code: str = Field(default="", description="Có thể để trống: server tự sinh mã Mermaid từ nodes/edges")
    ai_expansion_suggestions_vi: List[str] = Field(
        description="3-5 gợi ý từ LLM giúp người dùng bổ sung nhánh còn thiếu hoặc cải thiện cấu trúc Mind Map"
    )


# --- 2. 2D character branch ---
class ColorPalette(BaseModel):
    hair_color: str
    eye_color: str
    primary_outfit_color: str
    accent_color: str


class SuggestionChip(BaseModel):
    label_vi: str
    category: Literal["expression", "pose", "outfit", "accessory", "background"]
    prompt_delta_en: str
    recommended_workflow: Literal["variation_full", "expression_sheet", "local_inpaint"]


class CharacterDNAProfile(BaseModel):
    character_id: str
    character_archetype: str
    sketch_complexity_score: int = Field(ge=1, le=10)
    recommended_fidelity_mode: Literal["high_fidelity", "balanced", "creative_fix"]
    immutable_traits_en: List[str]
    mutable_outfit_en: str
    color_palette: ColorPalette
    contextual_suggestions: List[SuggestionChip]


# --- 3. Universal router envelope ---
InputCategory = Literal["character_or_object_sketch", "mindmap_or_flowchart_diagram", "character_concept_mindmap"]


class UniversalSketchAnalysisResponse(BaseModel):
    detected_input_category: InputCategory = Field(description="Phân loại tự động bức ảnh người dùng vừa tải lên")
    character_profile: Optional[CharacterDNAProfile] = None
    mindmap_graph: Optional[MindMapGraphStructure] = None


def to_mermaid(graph: MindMapGraphStructure) -> str:
    def q(s: str) -> str:
        return '"' + s.replace('"', "#quot;").replace("\n", "<br/>") + '"'

    shape = {
        "circle": "(({}))",
        "diamond": "{{{}}}",
        "rectangle": "[{}]",
        "pill": "([{}])",
        "triangle": "[/{}\\]",
        "cloud": ">{}]",
    }
    arrow = {"directed": "-->", "bidirectional": "<-->", "undirected_line": "---", "dashed": "-.->"}
    lines = ["flowchart TD"]
    for n in graph.nodes:
        lines.append(f"    {n.node_id}" + shape[n.shape_type].format(q(n.text_content_vi)))
    for e in graph.edges:
        label = f"|{q(e.relation_label_vi)}|" if e.relation_label_vi else ""
        lines.append(f"    {e.source_node_id} {arrow[e.arrow_type]}{label} {e.target_node_id}")
    for n in graph.nodes:
        lines.append(f"    style {n.node_id} fill:{n.suggested_color_hex}")
    return "\n".join(lines)


def validate_graph(graph: MindMapGraphStructure) -> MindMapGraphStructure:
    """Reject dangling edges / duplicate ids, regenerate Mermaid so it always matches the graph."""
    ids = [n.node_id for n in graph.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("node_id bị trùng")
    missing = {e.source_node_id for e in graph.edges} | {e.target_node_id for e in graph.edges}
    missing -= set(ids)
    if missing:
        raise ValueError(f"edge trỏ tới node không tồn tại: {sorted(missing)}")
    graph.mermaid_code = to_mermaid(graph)
    return graph
