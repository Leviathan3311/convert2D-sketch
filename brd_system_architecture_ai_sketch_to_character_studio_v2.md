# BRD & System Architecture v2.2 — AI Sketch-to-2D Studio (Hỗ trợ cả Nhân vật 2D & Mind Map / Sơ đồ Khối)

**Phiên bản:** 2.2 (Tích hợp Bộ định tuyến Dual-Engine: Character Diffusion Engine & Mind Map Vector/Hybrid Engine)

**Ngày cập nhật:** 29/10/2026

**Trạng thái:** Bản đặc tả Kiến trúc Kỹ thuật & Sản phẩm (Discovery / MVP Blueprint)

**Tên sản phẩm:** Sketch-to-2D Studio (Characters & Visual Mind Maps)

## 1. Tóm tắt điều hành & Định vị cạnh tranh

### 1.1 Ý tưởng cốt lõi

Xây dựng nền tảng web giúp người dùng biến **mọi bản phác thảo tay trên giấy** thành sản phẩm 2D chuyên nghiệp thông qua **Bộ định tuyến Thị giác Thông minh (VLM Router)**:

1. **Nếu đầu vào là Bản vẽ Nhân vật / Vật thể:** Chuyển thành nhân vật 2D hoàn chỉnh, khóa hồ sơ nhận diện (**Character DNA**) và tạo bộ biến thể (biểu cảm, tư thế, trang phục, sticker PNG tách nền) qua cụm GPU chạy [ComfyUI](https://github.com/Comfy-Org/ComfyUI).

2. **Nếu đầu vào là Sơ đồ Tư duy (Mind Map) / Flowchart (vẽ hình tam giác, tròn, vuông có ghi chữ bên trong và nối nhánh):**

   * **Chế độ Sơ đồ 2D Sắc nét (Vector Diagram / Mind Map Mode):** Chuyển bản vẽ tay nguệch ngoạc thành sơ đồ 2D Vector chuẩn xác 100% chữ tiếng Việt, cho phép kéo thả các khối tam giác/tròn, sửa chữ và xuất ảnh 2D/SVG/PDF qua [xyflow/xyflow](https://github.com/xyflow/xyflow), [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw), [mermaid-js/mermaid](https://github.com/mermaid-js/mermaid) hoặc [markmap/markmap](https://github.com/markmap/markmap).

   * **Chế độ Mind Map Minh họa (Visual Sketchnote / Illustrated Mind Map Mode):** Giữ cấu trúc sơ đồ và chữ tiếng Việt sắc nét bằng lớp Vector, đồng thời tự động gọi [ComfyUI](https://github.com/Comfy-Org/ComfyUI) sinh các icon/hình minh họa 2D gắn vào từng nhánh của Mind Map.

   * **Chế độ Mind Map Ý tưởng Nhân vật (Concept Mind Map → Character):** Nếu người dùng vẽ một Mind Map mô tả ý tưởng nhân vật (nhánh *Ngoại hình*, nhánh *Trang phục*, nhánh *Vũ khí*, nhánh *Tính cách*), VLM gom toàn bộ các nhánh để tổng hợp thành một nhân vật 2D hoàn chỉnh.

### 1.2 Tại sao không dùng chung 1 Model Tạo ảnh (Diffusion) cho cả Nhân vật và Mind Map?

| Tiêu chí | Đưa ảnh Mind Map vào Model tạo ảnh Pixel (Stable Diffusion / Chatbot tạo ảnh) | Kiến trúc Dual-Engine (VLM Graph Parser + Vector 2D + ComfyUI Icon) | 
| ----- | ----- | ----- | 
| **Độ chính xác chữ viết trong hình khối** | **Sai lỗi nặng:** Model tạo ảnh pixel vẽ chữ theo kiểu "hội họa", làm biến dạng dấu tiếng Việt hoặc biến chữ trong hình tam giác thành ký tự vô nghĩa. | **Chính xác 100%:** VLM đọc chữ viết tay (OCR) rồi render bằng font chữ Vector thật bên trong khối tam giác/tròn/vuông. | 
| **Khả năng chỉnh sửa sau khi tạo** | Ảnh phẳng (flat bitmap): muốn sửa 1 chữ trong hình tam giác hoặc kéo dịch 1 nhánh là phải tạo lại cả bức ảnh. | **Tương tác trực tiếp:** Mỗi hình tam giác, hình tròn và mũi tên là một Node độc lập trên Canvas; bấm đúp để sửa chữ, đổi màu khối hoặc kéo thả nhánh tùy ý. | 
| **Độ sắc nét của hình khối & mũi tên** | Cạnh hình tam giác dễ bị cong, mũi tên đứt đoạn hoặc nối nhầm nhánh khi sơ đồ có từ 5–10 khối trở lên. | Bảo toàn chính xác cấu trúc đồ thị (`Nodes` + `Edges`), căn chỉnh bố cục tự động (Auto-layout) phẳng phiu ở mọi độ phân giải. | 
| **Chi phí xử lý (Unit Cost)** | Tốn GPU sinh ảnh nặng nề (`~$0.015/lượt`) nhưng kết quả chữ lại bị lỗi. | Chỉ tốn **1 lệnh gọi VLM API (`~$0.0015/lượt`)** và render Vector trực tiếp trên trình duyệt (tiết kiệm 90% chi phí GPU). | 

## 2. Kiến trúc Hệ thống Tổng thể (Dual-Engine Architecture)

Kiến trúc v2.2 đặt **VLM Input Classifier & Router** ngay tại tầng Backend để tự động nhận diện loại bản vẽ và điều hướng sang đúng Engine xử lý:

### 2.1 Sơ đồ Kiến trúc Đầy đủ

```
[Client Tier: Next.js / React Web App]
  ├─ Upload Ảnh chụp giấy bất kỳ (Bản vẽ Nhân vật HOẶC Sơ đồ Mind Map / Khối hình học)
  ├─ Dual Workspace Canvas:
  │     ├─ (A) Character Studio Canvas (Preset Style, Thanh trượt Fidelity, Suggestion Chips)
  │     └─ (B) Interactive 2D Mind Map Canvas (Kéo thả khối tam giác/tròn, sửa chữ, gắn Icon 2D)
  └─ Export PNG tách nền / SVG Vector / PDF
        │
        │ HTTPS (REST API + Server-Sent Events / WebSocket)
        ▼
[Application & Orchestration Tier: FastAPI Backend]
  ├─ 1. Image Preprocessor (OpenCV khử bóng đổ giấy, làm phẳng nghiêng ảnh chụp)
  ├─ 2. VLM Intelligence Hub & Universal Router (Cloud API: Claude / Gemini / GPT)
  │     ├─ Phân loại tự động: [Character Sketch] vs. [Mind Map / Block Diagram]
  │     ├─ Nhánh Nhân vật: Trích xuất Character DNA + Biên dịch tham số ComfyUI
  │     └─ Nhánh Mind Map: Trích xuất cấu trúc Đồ thị (Nodes: hình dạng, màu, chữ + Edges: mũi tên)
  └─ 3. Project, Versioning & Credit Ledger Service
        │
        ├──────────────────────────────────────────────────────────┐
        ▼ (Nếu là Bản vẽ Nhân vật HOẶC cần vẽ Icon cho Mind Map)    ▼ (Nếu là Mind Map / Sơ đồ khối)
[GPU Inference Tier: ComfyUI Worker]                     [2D Vector & Diagram Engine]
  ├─ ComfyUI Headless + comfyui_controlnet_aux             ├─ Render Đồ thị Node/Edge (xyflow / React Flow)
  ├─ IPAdapter_plus + PuLID + StoryDiffusion               ├─ Chế độ Vẽ tay 2D đẹp (Excalidraw JSON)
  └─ BiRefNet (Tách nền PNG cho Nhân vật / Icon)           └─ Chế độ Sơ đồ Cây/Flowchart (Mermaid / Markmap)

```

### 2.2 Ma trận Công nghệ & GitHub Repositories Open Source

| Nhóm chức năng | Thành phần | Giải pháp / GitHub Repository | Vai trò Kỹ thuật trong Hệ thống | 
| ----- | ----- | ----- | ----- | 
| **Engine 1: Nhân vật & Icon 2D (Raster)** | Quản lý đồ thị sinh ảnh | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Chạy ngầm làm stateless GPU worker sinh nhân vật 2D và các icon minh họa cho Mind Map. | 
| **Engine 1: Nhân vật 2D** | Tiền xử lý nét vẽ tay | [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | Trích xuất `AnyLine`, `Manga2Anime_LineArt`, `PIDI` (Scribble), `HED` và `OpenPose`. | 
| **Engine 1: Nhân vật 2D** | Khóa nhận diện & Biểu cảm | [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/comfyui_ipadapter_plus), [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) & [HVision-NKU/StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) | Giữ nguyên khuôn mặt, trang phục và tạo bộ biểu cảm đồng nhất qua nhiều tư thế. | 
| **Engine 1: Nhân vật 2D** | Tách nền PNG trong suốt | [ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet) & [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) | Tách nền sắc nét cho nhân vật và các sticker/icon đính kèm vào nhánh Mind Map. | 
| **Engine 2: Mind Map & Sơ đồ Khối (Vector 2D)** | Canvas Sơ đồ Khối tương tác | [xyflow/xyflow](https://github.com/xyflow/xyflow) (React Flow) | Render các khối hình học (tam giác, tròn, thoi, chữ nhật) có chữ bên trong và dây nối; hỗ trợ kéo thả, sửa chữ và chèn ảnh 2D vào từng nút. | 
| **Engine 2: Mind Map 2D** | Phong cách 2D Sketchnote đẹp | [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw) | Biến sơ đồ vẽ tay nguệch ngoạc trên giấy thành bản vẽ 2D phong cách hand-drawn chuyên nghiệp, giữ đúng hình tam giác/tròn/mũi tên người dùng vẽ. | 
| **Engine 2: Mind Map 2D** | Sơ đồ Tư duy phân nhánh | [markmap/markmap](https://github.com/markmap/markmap) & [mermaid-js/mermaid](https://github.com/mermaid-js/mermaid) | Tự động dàn trang (auto-layout) cây tư duy và lưu đồ thuật toán từ dữ liệu do VLM đọc được. | 
| **Tầng Trí tuệ (Cả 2 Engine)** | Đọc chữ viết tay & Điều phối | Cloud VLM API hoặc [QwenLM/Qwen3-VL](https://github.com/qwenlm/qwen3-vl) + [PaddlePaddle/PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR) | Đọc chữ tiếng Việt viết tay trong các hình tam giác/tròn, nhận diện mũi tên liên kết và sinh JSON cấu trúc đồ thị. | 
| **Cầu nối Code** | Biên dịch ComfyUI sang Python | [pydn/ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension) & [Comfy-Org/comfy-python-sdk](https://github.com/Comfy-Org/comfy-python-sdk) | Chuyển workflow ComfyUI thành code Python để kết nối với FastAPI. | 

## 3. Pipeline Xử lý Chi tiết khi Đầu vào là Mind Map / Sơ đồ Khối có Chữ

Khi người dùng chụp một tờ giấy vẽ các hình tam giác, hình tròn, hình chữ nhật có ghi chữ bên trong và nối với nhau bằng các đường kẻ/mũi tên, hệ thống xử lý theo **3 chế độ đầu ra** tùy nhu cầu người dùng:

```
[Ảnh chụp giấy vẽ Mind Map: Hình tam giác ("Mục tiêu"), Hình tròn ("Ý tưởng A"), Mũi tên nối]
       │
       ▼
(Bước 1) VLM Graph Extractor (Đọc cấu trúc Hình học + Chữ viết tay + Liên kết)
       ├─ Trích xuất Nodes: id, shape ("triangle" | "circle" | "rectangle"), label_text, position (x, y), color
       └─ Trích xuất Edges: source_node_id ──► target_node_id, arrow_label, line_style
       │
       ├─────────────────────────────┬─────────────────────────────┐
       ▼                             ▼                             ▼
[Chế độ 1: Clean 2D Diagram]  [Chế độ 2: Visual Sketchnote] [Chế độ 3: Mind Map → Character]
Dựng lại đúng hình tam giác,   Giữ nguyên sơ đồ 2D Vector    Nếu các nhánh mô tả thuộc tính
hình tròn, mũi tên & chữ       sắc nét + Gọi ComfyUI sinh    nhân vật ("Tóc đỏ", "Giáp sắt"),
tiếng Việt sắc nét trên        các Icon/Hình minh họa 2D     gom toàn bộ nhánh thành Prompt
Excalidraw / React Flow.       nhỏ gắn cạnh từng khối.       để vẽ ra Nhân vật 2D hoàn chỉnh.

```

### 3.1 Chế độ 1: Chuyển Bản vẽ tay thành Sơ đồ 2D Chuẩn xác (`Clean 2D Vector Diagram`)

* **Cách hoạt động:** VLM đọc ảnh chụp tờ giấy và trả về cấu trúc JSON gồm danh sách `nodes` (lưu đúng hình dạng người dùng đã vẽ: `triangle`, `circle`, `diamond`, `rectangle`, tọa độ tương đối `(x, y)` trên trang giấy, và nội dung chữ bên trong) cùng danh sách `edges` (các đường nối/mũi tên).

* **Hiển thị & Chỉnh sửa:** Frontend dùng [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw) hoặc [xyflow/xyflow](https://github.com/xyflow/xyflow) để vẽ lại toàn bộ các hình tam giác, hình tròn và chữ viết đó thành đồ họa 2D phẳng, sạch đẹp, thẳng hàng.

* **Ứng dụng LLM Suggestion cho Mind Map:** Ngay sau khi số hóa sơ đồ lên màn hình, LLM phân tích nội dung các nhánh và đưa ra gợi ý:

  * *"Mở rộng thêm 3 nhánh con cho khối tam giác \[Chiến lược Marketing\]"*

  * *"Tự động căn chỉnh lại bố cục (Auto-Layout) cho cân đối"*

  * *"Đổi màu các nhánh theo nhóm chủ đề"*

### 3.2 Chế độ 2: Mind Map Minh họa 2D (`Visual Sketchnote / Illustrated Mind Map`)

* Nếu người dùng muốn biến một bản vẽ Mind Map khô khan thành một **bản Infographic / Sketchnote 2D sinh động**:

  1. Các hình khối (tam giác, tròn), đường nối và chữ tiếng Việt vẫn được vẽ bằng **Vector 2D** để chữ tuyệt đối không bị lỗi font.

  2. Với mỗi nhánh chính trên Mind Map, LLM tự động tóm tắt thành một từ khóa hình ảnh (ví dụ: nhánh *"Tăng trưởng doanh thu"* → icon `"2D flat illustration of a rocket launching with gold coins"`) và đẩy vào hàng đợi [ComfyUI](https://github.com/Comfy-Org/ComfyUI) + [BiRefNet](https://github.com/zhengpeng7/birefnet) để sinh các sticker 2D tách nền gắn ngay cạnh hình tam giác/hình tròn đó.

### 3.3 Chế độ 3: Dùng Mind Map làm Bản thiết kế Nhân vật (`Concept Mind Map → 2D Character`)

* Rất nhiều người không biết vẽ hình người nhưng lại thích phác thảo ý tưởng nhân vật dưới dạng **Mind Map** (ví dụ: vẽ một hình tròn ở giữa ghi *"Nhân vật chính"*, nối ra hình tam giác 1 ghi *"Đội mũ phi hành gia nứt kính"*, nối ra hình tam giác 2 ghi *"Mặc áo choàng rách màu cam"*, nối ra hình vuông 3 ghi *"Cầm cờ lê phát sáng"*).

* Khi người dùng chọn *"Tạo nhân vật từ Mind Map này"*, tầng LLM gom toàn bộ các nút (`nodes`) trên sơ đồ tư duy đó thành hồ sơ `CharacterDNAProfile` hoàn chỉnh và đẩy sang `Workflow 1 (sketch_to_2d_v1)` của ComfyUI để vẽ ra nhân vật 2D đúng 100% các nhánh đã ghi chú.

## 4. Cấu trúc JSON Schema (Pydantic) Hợp nhất cho cả Nhân vật & Mind Map

Bạn có thể dùng trực tiếp bộ Pydantic Schema dưới đây trong FastAPI để VLM tự động xử lý cả 2 loại đầu vào:

```
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# --- 1. SCHEMA CHO NHÁNH MIND MAP / SƠ ĐỒ KHỐI ---
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
        description="Prompt tiếng Anh ngắn để ComfyUI vẽ icon 2D minh họa cho nút này (nếu bật chế độ Illustrated Mind Map)"
    )

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
    mermaid_code: str = Field(description="Mã Mermaid.js tương đương để render nhanh hoặc xuất Markdown")
    ai_expansion_suggestions_vi: List[str] = Field(
        description="3-5 gợi ý từ LLM giúp người dùng bổ sung nhánh còn thiếu hoặc cải thiện cấu trúc Mind Map"
    )

# --- 2. SCHEMA CHO NHÁNH NHÂN VẬT 2D ---
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

# --- 3. ENVELOPE PHÂN LUỒNG TỔNG (UNIVERSAL ROUTER RESPONSE) ---
class UniversalSketchAnalysisResponse(BaseModel):
    detected_input_category: Literal[
        "character_or_object_sketch",
        "mindmap_or_flowchart_diagram",
        "character_concept_mindmap"
    ] = Field(description="Phân loại tự động bức ảnh người dùng vừa tải lên")
    character_profile: Optional[CharacterDNAProfile] = None
    mindmap_graph: Optional[MindMapGraphStructure] = None

```

## 5. Chi tiết Các Workflow Xử lý trong Hệ thống

### 5.1 Nhánh Engine 1 (ComfyUI GPU Worker — Cho Nhân vật & Icon Minh họa)

1. **Workflow 1 (`sketch_to_2d_v1`):** Dùng [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) (`AnyLine` / `HED` / `PIDI_Scribble`) + Base Checkpoint + Style LoRA để biến nét vẽ nhân vật thành ảnh 2D.

2. **Workflow 2 (`character_variation_v1`):** Dùng [ComfyUI_IPAdapter_plus](https://github.com/cubiq/comfyui_ipadapter_plus) + [PuLID](https://github.com/ToTheBeginning/PuLID) + `ControlNet OpenPose` để đổi tư thế/trang phục mà giữ nguyên khuôn mặt và nhận diện.

3. **Workflow 3 (`story_expression_pack_v1`):** Dùng [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) + [BiRefNet](https://github.com/zhengpeng7/birefnet) để sinh bộ 4–6 sticker biểu cảm tách nền trong 1 lần chạy.

4. **Workflow 4 (`inpaint_edit_v1`):** Khoanh vùng Mask để sửa chi tiết cục bộ trên nhân vật.

### 5.2 Nhánh Engine 2 (Vector & Graph Renderer — Cho Mind Map / Sơ đồ Khối)

1. **Workflow 5 (`mindmap_to_vector_2d_v1`):**

   * Nhận `MindMapGraphStructure` từ VLM API.

   * Ánh xạ từng `MindMapNode` (giữ đúng `shape_type`: hình tam giác, hình tròn, hình chữ nhật, hình thoi và tọa độ `(x_norm, y_norm)`) sang định dạng phần tử của [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw) hoặc [xyflow/xyflow](https://github.com/xyflow/xyflow).

   * Cho phép người dùng bấm 1 nút để chuyển đổi qua lại giữa **Bố cục giữ nguyên vị trí trên giấy** và **Bố cục tự động căn chỉnh đẹp (Dagre / D3 Auto-Layout)**.

2. **Workflow 6 (`mindmap_illustrated_sketchnote_v1`):**

   * Chạy `Workflow 5` để dựng khung sơ đồ và chữ Vector sắc nét.

   * Gửi batch các `icon_prompt_en` của các nút chính sang ComfyUI (sinh ở độ phân giải nhỏ `512x512` với SDXL Lightning/Turbo trong 2 giây) + tách nền qua [BiRefNet](https://github.com/zhengpeng7/birefnet) rồi tự động đính vào góc các khối tam giác/hình tròn trên Canvas.

## 6. Thiết kế Cơ sở dữ liệu (PostgreSQL Schema)

| Tên bảng | Các trường khóa & dữ liệu chính | Mục đích nghiệp vụ | 
| ----- | ----- | ----- | 
| `users` | `id`, `email`, `tier`, `credit_balance`, `created_at` | Quản lý tài khoản, gói cước và số dư credit. | 
| `projects` | `id`, `user_id`, `title`, `project_type` (`character_2d`, `mindmap_2d`, `hybrid_sketchnote`), `original_sketch_asset_id`, `status` | Hỗ trợ cả dự án tạo nhân vật 2D lẫn dự án số hóa Mind Map 2D. | 
| `character_profiles` | `id`, `project_id`, `canonical_asset_id`, `dna_json` (JSONB), `approved_reference_ids` (UUID\[\]), `style_preset_id` | Lưu trữ `CharacterDNAProfile` cho các dự án nhân vật. | 
| `mindmap_graphs` | `id`, `project_id`, `version`, `graph_json` (JSONB lưu `MindMapGraphStructure`), `mermaid_source` (TEXT), `excalidraw_scene` (JSONB) | Lưu trữ trạng thái đồ thị 2D của Mind Map để người dùng sửa chữ, thêm nhánh và quay lại lịch sử phiên bản. | 
| `assets` | `id`, `project_id`, `parent_asset_id`, `type` (`sketch_raw`, `sketch_clean`, `canonical`, `variation`, `sticker_png`, `mindmap_svg`), `s3_key`, `seed` | Quản lý toàn bộ file ảnh raster và vector xuất ra. | 
| `generation_jobs` | `id`, `project_id`, `engine_type` (`vlm_only_vector`, `comfy_gpu`, `hybrid`), `llm_tokens_cost_usd`, `gpu_seconds`, `gpu_cost_usd` | Hạch toán riêng chi phí job chỉ dùng VLM (Mind Map Vector) và job dùng GPU (Nhân vật 2D). | 
| `credits_ledger` | `id`, `user_id`, `job_id`, `amount_delta`, `transaction_type` | Trừ credit linh hoạt (ví dụ: chuyển Mind Map sang Vector tốn 1 credit; tạo bộ Nhân vật 2D tốn 5 credits). | 

## 7. Chiến lược Hạ tầng & Lộ trình Triển khai

* **Giai đoạn MVP (Cloud VLM API + 1 GPU Worker + Vector Frontend):**

  * Dùng Cloud VLM API (Claude Sonnet/Haiku hoặc Gemini Flash) làm cổng `UniversalSketchAnalysisResponse`.

  * Nhánh **Mind Map 2D** có thể hoàn thiện cực nhanh trên Frontend Next.js nhờ nhúng trực tiếp thư viện MIT License [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw) hoặc [xyflow/xyflow](https://github.com/xyflow/xyflow) mà không tốn bất kỳ giây GPU nào.

  * Nhánh **Nhân vật 2D** sử dụng toàn bộ tài nguyên GPU Worker với [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux), [ComfyUI_IPAdapter_plus](https://github.com/cubiq/comfyui_ipadapter_plus) và [BiRefNet](https://github.com/zhengpeng7/birefnet).

* **Giai đoạn Scale (Self-host VLM):**

  * Khi số lượng ảnh chụp Mind Map và Sketch vượt 10.000 lượt/ngày, triển khai [QwenLM/Qwen3-VL](https://github.com/qwenlm/qwen3-vl) trên [vllm-project/vllm](https://github.com/vllm-project/vllm) để đọc cả chữ viết tay tiếng Việt trong các khối hình học lẫn phân tích nét vẽ nhân vật với chi phí nội bộ tối ưu nhất.


## 8. Phân tích Chi phí Vận hành & Unit Economics Thực tế (Cập nhật 2026)

### 8.1 Đơn giá Hạ tầng Cơ sở (Cloud API + Serverless GPU + Storage)

| Hạng mục hạ tầng | Nhà cung cấp & Cấu hình đề xuất | Đơn giá tham chiếu | Ghi chú tối ưu chi phí |
| :--- | :--- | :--- | :--- |
| **Tầng VLM / LLM API** | [Claude Haiku 4.5 / 5.5](https://www.anthropic.com/claude/haiku) hoặc [Claude Sonnet](https://www.anthropic.com/claude-sonnet-5-5) ([Bảng giá Claude API](https://platform.claude.com/docs/en/about-claude/pricing)) | **$0.0005 – $0.002 / lượt gọi** (~13đ – 50đ VNĐ) | Chỉ trả tiền khi có ảnh gửi lên; 0đ chi phí máy chờ (idle). |
| **GPU Sinh ảnh (Serverless — MVP)** | NVIDIA L4 (24GB VRAM) trên [Modal](https://modal.com/pricing) hoặc RTX 4090 (24GB VRAM) trên [RunPod Serverless](https://www.runpod.io/product/serverless) | **L4: ~$0.80/giờ** ($0.000222/giây) / **RTX 4090 Serverless: ~$1.10/giờ** ($0.000306/giây) | Tính tiền theo từng giây chạy thực tế; [Modal](https://modal.com/pricing) tặng sẵn **$30/tháng** credit miễn phí ở gói Starter. |
| **GPU Sinh ảnh (Dedicated Pod — Scale)** | Thuê cố định On-Demand Pod [RTX 4090 trên RunPod](https://deploybase.ai/articles/rtx-4090-runpod) | **$0.34/giờ (~$244/tháng chạy 24/7)** | Rẻ hơn 70% mỗi giây so với Serverless khi lưu lượng ổn định trên 3.000 ảnh/ngày. |
| **Lưu trữ Ảnh (Object Storage)** | [Cloudflare R2](https://developers.cloudflare.com/r2/pricing/) | **Miễn phí 10 GB đầu**, vượt mức tính **$0.015 / GB-tháng** | **Miễn phí 100% băng thông tải ảnh ra (Zero Egress Fee)**, tiết kiệm lớn so với AWS S3. |
| **Web App, API & Database** | Vercel + Railway / Render (FastAPI + PostgreSQL + Redis) | **$0 – $25 / tháng** ở giai đoạn MVP | Gói Free/Hobby đủ sức chịu tải 500–1.000 người dùng thử nghiệm đầu tiên. |

### 8.2 Chi phí Giá vốn (COGS) trên Từng Tác vụ của Người dùng

| Loại tác vụ người dùng thực hiện | Luồng xử lý kỹ thuật | Thời gian GPU | Chi phí VLM API | Chi phí GPU | Tổng Giá vốn (USD) | Tổng Giá vốn (VNĐ) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Chuyển ảnh Mind Map sang Sơ đồ 2D Vector** | VLM đọc chữ + cấu trúc khối → Render trực tiếp trên trình duyệt qua [excalidraw](https://github.com/excalidraw/excalidraw) / [xyflow](https://github.com/xyflow/xyflow) | **0 giây** | ~$0.0015 | **$0.0000** | **~$0.0015** | **~38đ / sơ đồ** |
| **2. Mind Map Minh họa (Sketchnote 2D có 4 Icon)** | VLM đọc cấu trúc + ComfyUI sinh batch 4 icon nhỏ (`512x512` SDXL Lightning) + tách nền [BiRefNet](https://github.com/zhengpeng7/birefnet) | ~4 giây | ~$0.0020 | ~$0.0012 | **~$0.0032** | **~80đ / sơ đồ** |
| **3. Tạo Nhân vật 2D gốc từ Sketch (Batch 4 phương án)** | OpenCV làm sạch + VLM trích xuất `CharacterDNA` + ComfyUI sinh 4 ảnh `1024x1024` | ~12–15 giây | ~$0.0015 | ~$0.0040 | **~$0.0055** | **~140đ / lượt (4 ảnh)** |
| **4. Tạo 1 Biến thể Tư thế / Trang phục + Tách nền PNG** | LLM biên dịch JSON + ComfyUI (`IP-Adapter` + `PuLID`) + [BiRefNet](https://github.com/zhengpeng7/birefnet) | ~6–8 giây | ~$0.0010 | ~$0.0022 | **~$0.0032** | **~80đ / ảnh PNG** |
| **5. Xuất trọn bộ 1 Character Pack hoàn chỉnh (kể cả thử sai)** | Gồm 2 lần tạo nhân vật gốc + 6 lần sinh biến thể/biểu cảm (tổng 14 ảnh sinh ra để lấy 5–6 ảnh ưng ý) | ~65 giây | ~$0.0090 | ~$0.0190 | **~$0.0280** | **~700đ / bộ nhân vật** |

### 8.3 Dự phóng Chi phí Vận hành Hàng tháng theo 3 Giai đoạn

| Giai đoạn | Quy mô người dùng & Sản lượng | Cấu hình Hạ tầng đề xuất | Tổng Chi phí Vận hành / tháng | Biên lợi nhuận gộp (Gross Margin) dự kiến |
| :--- | :--- | :--- | :--- | :--- |
| **Giai đoạn 1: Discovery & Private Beta (Tháng 1–2)** | **50–200 users** (~500 bộ nhân vật + 1.000 Mind Map / tháng) | Serverless GPU ([Modal](https://modal.com/pricing) tận dụng $30 free credit) + [Cloudflare R2](https://developers.cloudflare.com/r2/pricing/) Free Tier + Cloud VLM API | **$10 – $25 / tháng** (~250.000đ – 650.000đ/tháng) | Rủi ro tài chính gần như bằng 0 trong giai đoạn kiểm chứng thị trường. |
| **Giai đoạn 2: Ra mắt MVP Thương mại (Tháng 3–6)** | **1.000 active users** — giả sử 100 người trả phí gói Creator `129.000đ/tháng` (~12,9 triệu đ doanh thu); tổng tạo ~3.000 bộ nhân vật + 5.000 Mind Map | Serverless GPU (Modal / [RunPod Serverless](https://www.runpod.io/product/serverless)) + Cloud VLM API + Backend/DB ($25) + R2 ($5) | **~$135 / tháng** (~3,4 triệu VNĐ/tháng) | **~73% Gross Margin** (Lợi nhuận gộp ~9,5 triệu VNĐ/tháng chỉ với 100 khách trả phí). |
| **Giai đoạn 3: Tăng trưởng / Scale (Trên 5.000 users)** | **500–1.000 khách trả phí** (~65–130 triệu đ doanh thu/tháng); tổng tạo > 20.000 bộ nhân vật/tháng | Chuyển sang 1–2 Dedicated Pod [RTX 4090 ($244/pod)](https://deploybase.ai/articles/rtx-4090-runpod) + tự host [Qwen3-VL](https://github.com/qwenlm/qwen3-vl) hoặc Batch VLM API | **~$400 – $650 / tháng** (~10 – 16,5 triệu VNĐ/tháng) | **80% – 85% Gross Margin** nhờ tận dụng tối đa công suất GPU cố định. |


## 9. Kiến trúc Chạy Cục bộ (Local Edition) cho MacBook M1 Pro + Gói Claude Pro Sẵn có (0đ Chi phí Phát sinh)

### 9.1 Giải bài toán 2 giới hạn thực tế trên MacBook M1 Pro & Gói Claude Pro
1. **Về gói Claude Pro ($20/tháng trên Web/Desktop):** Gói Claude Pro không bao gồm API Key trả theo token (`api.anthropic.com`). Tuy nhiên, bạn có thể **tận dụng 100% gói Claude Pro sẵn có với giá 0đ** bằng cách dùng **Claude Desktop / Claude Code kết nối qua giao thức MCP (Model Context Protocol)** thông qua [Comfy-Org/comfy-mcp](https://github.com/Comfy-Org/comfy-mcp) và [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk). Khi đó, chính ứng dụng Claude Desktop trên máy Mac của bạn đóng vai trò là **Bộ não VLM/LLM Orchestrator** điều khiển trực tiếp ComfyUI và Web UI tại máy local!
2. **Về giới hạn RAM Gộp (Unified Memory 16GB / 32GB) của MacBook M1 Pro:**
   * Tuyệt đối không bật Docker nặng (PostgreSQL, Redis, MinIO) ngốn 3–4GB RAM nền. Thay toàn bộ bằng **SQLite + Local Folder + `asyncio.Queue` trong bộ nhớ**.
   * Không chạy model SD3.5 Medium FP16 quá nặng trên M1 Pro (sẽ mất 2–3 phút/ảnh). Thay bằng **SDXL Lightning (4–8 steps)** hoặc **DreamShaper / Anime SD 1.5 + LCM LoRA** trên nhân đồ họa **Apple Metal (`mps`)** để rút thời gian sinh ảnh trên M1 Pro xuống chỉ còn **8 – 18 giây/ảnh**.

### 9.2 Sơ đồ Kiến trúc Local trên MacBook M1 Pro

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                     MACBOOK M1 PRO (Apple Silicon MPS)                     │
│                                                                            │
│  [1. Bộ não VLM/LLM: Gói Claude Pro sẵn có (0đ tốn thêm)]                  │
│   ├─ Giao diện điều phối: Claude Desktop App / Claude Code                 │
│   ├─ Đọc ảnh Sketch & ảnh Mind Map tiếng Việt cực chuẩn                    │
│   └─ Giao tiếp qua giao thức MCP (Model Context Protocol)                  │
│         │                                                                  │
│         ▼ (Gọi MCP Tools cục bộ)                                           │
│  [2. Tầng Điều phối & MCP Bridge: FastAPI + FastMCP (localhost:8000)]      │
│   ├─ MCP Tool 1: `preprocess_and_clean_sketch()` (OpenCV lọc nền giấy)     │
│   ├─ MCP Tool 2: `dispatch_character_workflow()` (Gửi JSON sang ComfyUI)   │
│   ├─ MCP Tool 3: `render_interactive_mindmap()` (Đẩy đồ thị ra Web UI)     │
│   └─ Lưu trữ siêu nhẹ: SQLite (`studio_local.db`) + Thư mục `./storage/`   │
│         │                                       │                          │
│         ▼ (HTTP / WebSocket: localhost:8188)    ▼ (WebSocket: port 3000)   │
│  [3. Engine Nhân vật: ComfyUI MPS]       [4. Engine Mind Map & Web Studio] │
│   ├─ Chạy cờ: `--force-fp16` trên Metal   ├─ Next.js UI (localhost:3000)   │
│   ├─ Base: SDXL Lightning 4-8 steps       ├─ Nhúng Excalidraw & React Flow │
│   ├─ ControlNet Aux (AnyLine / Scribble)  └─ Sửa chữ, kéo thả khối tam     │
│   ├─ IP-Adapter Plus (Khóa nhân vật)         giác/tròn & tải PNG/SVG       │
│   └─ BiRefNet-lite / RMBG (Tách nền PNG)                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Bảng Quy đổi Kiến trúc từ Cloud Production sang Local MacBook M1 Pro

| Tầng Kiến trúc | Bản Cloud Production (Khi thương mại hóa) | Bản Local MacBook M1 Pro (Chạy thử nghiệm ngay — 0đ) | Mức tiêu thụ RAM trên M1 Pro |
| :--- | :--- | :--- | :--- |
| **Tầng Trí tuệ VLM / LLM** | Gọi REST API trả phí theo token (`api.anthropic.com`) | Dùng trực tiếp **Claude Desktop / Claude Code (Gói Claude Pro sẵn có)** nối qua [Comfy-Org/comfy-mcp](https://github.com/Comfy-Org/comfy-mcp) & [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | **~300 MB RAM** (vì mô hình Claude chạy trên server Anthropic, không ăn VRAM của Mac) |
| **Web Frontend & Mind Map 2D** | Next.js trên Vercel | Next.js chạy `localhost:3000` tích hợp [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw) & [xyflow/xyflow](https://github.com/xyflow/xyflow) | **~400 MB RAM** |
| **Backend & Hàng đợi (Queue)** | FastAPI + Redis + Celery Worker (Docker) | FastAPI chạy trực tiếp trong `uv` / `venv` + `asyncio.Queue` nội bộ (Không dùng Docker) | **~120 MB RAM** |
| **Cơ sở dữ liệu & Lưu trữ File** | PostgreSQL + Cloudflare R2 / AWS S3 | **SQLite** (1 file `local.db`) + Lưu ảnh trực tiếp vào ổ cứng SSD của Mac (`./local_assets/`) | **~10 MB RAM** |
| **Engine Sinh ảnh (ComfyUI)** | GPU Server L4 / RTX 4090 chạy SD3.5 / SDXL 30 steps + PuLID + StoryDiffusion | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) chạy native trên **Apple Silicon MPS** (`python main.py --force-fp16`) | **~6.5 GB – 9.5 GB Unified Memory** (An toàn cho cả bản M1 Pro 16GB RAM) |
| **Bộ Model tối ưu tốc độ cho M1 Pro** | Checkpoint FP16 nặng (6.5GB – 12GB) | **SDXL-Lightning (4-step / 8-step)** hoặc **DreamShaper 8 (SD1.5) + LCM LoRA** + [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) + [ComfyUI_IPAdapter_plus](https://github.com/cubiq/comfyui_ipadapter_plus) | Thời gian sinh 1 ảnh trên M1 Pro chỉ mất **8 – 18 giây** (thay vì 90–120 giây nếu chạy SDXL chuẩn 30 steps) |
| **Tách nền PNG trong suốt** | BiRefNet Full HR | [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) hoặc [ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet) bản nhẹ (`BiRefNet_lite`) | **~1.5 giây / ảnh** trên M1 Pro |

### 9.4 Cấu hình Thông số ComfyUI Chuẩn để không bị Tràn RAM (Swap/OOM) trên M1 Pro
Khi khởi động [ComfyUI](https://github.com/Comfy-Org/ComfyUI) trên MacBook M1 Pro, bạn nên cấu hình các tham số sau:
* **Lệnh khởi chạy cho máy M1 Pro 16GB RAM:** `python main.py --force-fp16 --lowvram --preview-method auto`
* **Lệnh khởi chạy cho máy M1 Pro 32GB RAM:** `python main.py --force-fp16 --preview-method auto`
* **Độ phân giải sinh ảnh khuyến nghị trên M1 Pro:**
  * Nếu dùng họ **SD 1.5 (Anime / Cartoon / Chibi):** Sinh ở `512x512` hoặc `512x768` (~6 giây/ảnh), sau đó dùng node `Upscale Model (RealESRGAN_x2)` lên `1024x1536` (~2 giây) — nhanh gấp 4 lần so với vẽ trực tiếp ở độ phân giải cao.
  * Nếu dùng họ **SDXL Lightning:** Sinh ở `1024x1024` với `steps = 6`, `cfg = 1.5 – 2.0`, `sampler = euler`, `scheduler = sgm_uniform` (~14–18 giây/ảnh).


## 10. Kiến trúc Hybrid: Google Colab (GPU T4/L4) + MacBook M1 Pro + Gói Claude Pro (Nhẹ máy, 0đ Chi phí)

Nếu bạn không muốn MacBook M1 Pro bị nóng máy, tốn 30–50GB ổ cứng SSD chứa model AI hoặc bị tràn RAM, bạn hoàn toàn có thể **đẩy riêng phần nặng nhất (ComfyUI + GPU) lên Google Colab**, còn **MacBook M1 Pro chỉ chạy phần điều phối nhẹ (Claude Pro MCP + Web UI Mind Map/Studio)** và nối với nhau qua đường hầm bảo mật [cloudflare/cloudflared](https://github.com/cloudflare/cloudflared).

### 10.1 Sơ đồ Kiến trúc Hybrid (MacBook M1 Pro ↔ Google Colab GPU)

```text
┌─────────────────────────────────────────────┐       HTTPS Tunnel (Cloudflared)      ┌──────────────────────────────────────────────┐
│        MACBOOK M1 PRO (Rất mát & nhẹ)       │    https://xxxx.trycloudflare.com     │        GOOGLE COLAB (NVIDIA T4 15GB VRAM)    │
│                                             │ ════════════════════════════════════► │                                              │
│  [1. Gói Claude Pro sẵn có (0đ API)]        │                                       │  [4. ComfyUI Headless Worker (CUDA)]         │
│   ├─ Claude Desktop / Claude Code           │                                       │   ├─ Chạy trên card NVIDIA T4 (15GB VRAM)    │
│   ├─ Nhìn ảnh Sketch & Mind Map tiếng Việt  │ ◄════════════════════════════════════ │   ├─ Tương thích 100% CUDA / xformers        │
│   └─ Gọi MCP Tools trên máy Mac             │    Trả về ảnh PNG tách nền (2–5s)     │   ├─ Chạy tốt SDXL + ControlNet + IP-Adapter │
│         │                                   │                                       │   ├─ Chạy PuLID + StoryDiffusion + BiRefNet  │
│         ▼                                   │                                       │   └─ Mở cổng 8188 qua `cloudflared`          │
│  [2. MCP Bridge + FastAPI (localhost:8000)] │                                       │         ▲                                    │
│   ├─ Tiền xử lý ảnh nhẹ (OpenCV)            │                                       │         │ (Mount thư mục Model cố định)      │
│   ├─ Biến môi trường:                       │                                       │  [5. Google Drive Cá nhân]                   │
│   │  `COMFY_URL = https://xxxx...`          │                                       │   └─ Lưu sẵn Checkpoints / LoRAs để không    │
│   └─ SQLite (`studio_local.db`)             │                                       │      phải tải lại 15GB mỗi lần bật Colab     │
│         │                                   │                                       └──────────────────────────────────────────────┘
│         ▼                                   │
│  [3. Next.js Web Studio (localhost:3000)]   │
│   ├─ Vẽ Mind Map 2D (Excalidraw / Xyflow)   │
│   └─ Hiển thị bộ Nhân vật 2D từ Colab về    │
└─────────────────────────────────────────────┘
```

### 10.2 So sánh Chạy ComfyUI trên MacBook M1 Pro vs. Đẩy lên Google Colab

| Tiêu chí kỹ thuật | Chạy 100% trên MacBook M1 Pro (Apple MPS) | Chạy Hybrid: MacBook (Claude Pro + UI) + Google Colab (GPU T4) |
| :--- | :--- | :--- |
| **Dung lượng ổ cứng Mac bị chiếm** | Tốn **25 GB – 50 GB SSD** trên máy Mac để lưu Checkpoint, ControlNet, IP-Adapter, BiRefNet. | **0 GB trên máy Mac** (toàn bộ model tải thẳng vào ổ ảo của Colab hoặc lưu trên Google Drive). |
| **Mức chiếm RAM trên MacBook** | Tốn **8 GB – 11 GB RAM**, quạt quay mạnh khi sinh liên tục nhiều ảnh. | Chỉ tốn **< 500 MB RAM** trên máy Mac (máy mát lạnh như đang lướt web bình thường). |
| **Tốc độ sinh ảnh & Tương thích CUDA** | 8–18 giây/ảnh (với model rút gọn Lightning); một số custom node yêu cầu nhân `CUDA` hoặc `insightface` đôi khi kén chip Apple M1. | Card **NVIDIA Tesla T4 (15GB VRAM CUDA)** tương thích 100% mọi repo ([PuLID](https://github.com/ToTheBeginning/PuLID), [InstantID](https://github.com/instantx-research/instantid), [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion)) mà không lo lỗi thư viện Mac. |
| **Điểm bất tiện cần lưu ý** | Bật lên là chạy ngay lập tức (0 giây chờ khởi động). | Mỗi lần mở phiên Colab mới mất **~2–3 phút** chạy cell khởi động và lấy link `trycloudflare.com` dán vào file `.env` trên máy Mac; Colab Free sẽ ngắt sau ~60–90 phút nếu treo máy không dùng. |

### 10.3 Quy trình 3 bước kết nối Google Colab với gói Claude Pro trên máy Mac
1. **Bước 1 — Khởi chạy ComfyUI kèm Cloudflare Tunnel trên Google Colab:**
   * Sử dụng notebook mẫu từ [comfyui_colab_with_manager.ipynb](https://colab.research.google.com/github/ltdrdata/ComfyUI-Manager/blob/main/notebooks/comfyui_colab_with_manager.ipynb) hoặc [nazdridoy/ComfyUI-On-Colab](https://github.com/nazdridoy/ComfyUI-On-Colab) tích hợp sẵn [cloudflare/cloudflared](https://github.com/cloudflare/cloudflared).
   * Khi chạy cell cuối trên Colab, `cloudflared` sẽ in ra một đường dẫn công khai dạng: `https://random-words-here.trycloudflare.com`.
2. **Bước 2 — Trỏ MCP Server trên MacBook vào đường dẫn Colab:**
   * Trên MacBook M1 Pro, bạn cài [Comfy-Org/comfy-mcp](https://github.com/Comfy-Org/comfy-mcp) hoặc chạy FastAPI MCP Server (dùng [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) + [Comfy-Org/comfy-python-sdk](https://github.com/Comfy-Org/comfy-python-sdk)) và đặt biến môi trường `COMFY_API_URL="https://random-words-here.trycloudflare.com"`.
3. **Bước 3 — Dùng Claude Desktop (Claude Pro) điều khiển từ xa:**
   * Bạn gửi ảnh vẽ tay Nhân vật hoặc ảnh chụp Mind Map vào cửa sổ **Claude Desktop** trên máy Mac.
   * Nếu là **Mind Map**: Claude Pro đọc ảnh và vẽ ngay sơ đồ 2D Vector trên trình duyệt máy Mac (`localhost:3000`) bằng [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw) hoặc [xyflow/xyflow](https://github.com/xyflow/xyflow) trong 2 giây mà không cần gọi lên Colab.
   * Nếu là **Nhân vật 2D**: Claude Pro tự động biên dịch JSON `CharacterDNA` và bắn request qua đường hầm `trycloudflare.com` lên GPU T4 của Google Colab để vẽ ảnh, tách nền bằng [BiRefNet](https://github.com/zhengpeng7/birefnet) rồi tải ảnh PNG thành phẩm về máy Mac cho bạn!