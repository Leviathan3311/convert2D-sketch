# Sketch-to-2D Studio — bản MVP (Colab + Claude Desktop)

MVP của BRD v2.2 theo kiến trúc Hybrid ở §10: **Claude Desktop (gói Claude Pro) làm VLM/LLM** qua MCP, **ComfyUI chạy trên GPU T4 của Google Colab** qua `cloudflared`, MacBook chỉ chạy phần nhẹ (MCP bridge + SQLite + web studio).

```
Claude Desktop ──MCP stdio──► studio-mcp (Mac) ──HTTPS trycloudflare──► ComfyUI (Colab T4)
  (nhìn ảnh, phân loại,         ├─ OpenCV làm sạch giấy                  DreamShaper 8 + ControlNet
   đọc chữ, viết JSON)          ├─ SQLite + storage/                     lineart / scribble
                                └─ Web studio http://127.0.0.1:8765  ◄── React Flow (mind map vector)
```

## Cài đặt

```bash
cd convert2D-sketch
uv sync
```

(Tuỳ chọn) Đăng ký MCP server `sketch-studio` cho Claude Desktop trong
`~/Library/Application Support/Claude/claude_desktop_config.json` theo mẫu
[claude_desktop_config.example.json](../claude_desktop_config.example.json).
**Tắt hẳn Claude Desktop (⌘Q) rồi mở lại** để nó nạp server.

## Mỗi lần dùng

1. Mở [colab/comfyui_worker.ipynb](../colab/comfyui_worker.ipynb) trên Colab (File → Upload notebook), chọn GPU T4, Run all.
   Lần đầu tải ~3.6 GB model vào Google Drive, các lần sau chỉ ~2–3 phút.
2. Copy link `https://xxxx.trycloudflare.com` ở cell cuối.
3. Bỏ ảnh chụp giấy vào `inbox/` rồi chat với Claude Desktop, ví dụ:
   - *"Đặt comfy url là https://xxxx.trycloudflare.com"*
   - *"Mở bản vẽ mindmap.jpg trong sketch studio và số hoá thành sơ đồ"*
   - *"Mở hero.jpg, tạo nhân vật 2D phong cách chibi"* → *"lấy kết quả job"* → *"tạo biến thể cười lớn từ ast_xxxx"* → *"tách nền ast_yyyy"*
   - *"Thêm icon minh hoạ cho các nhánh chính rồi sinh icon"* (chế độ Sketchnote)
   - *"Tạo nhân vật từ mind map này"* (chế độ Concept Mind Map → Character)
4. Web studio mở tự động: kéo thả khối, sửa chữ, đổi hình/màu, nối nhánh, tự căn bố cục, xuất PNG/SVG/JSON/Mermaid.
   Muốn mở web khi không có Claude Desktop: `uv run studio-web`.

## Dùng chỉ bằng web (không cần Claude Desktop) — Qwen tự host trên Colab

1. Colab: bật `ENABLE_LLM` (mặc định), Run all → cell Ollama in ra `COMFY_URL` và `LLM_URL`.
   Model tự chọn theo VRAM: T4 → `qwen3.5:9b-q4_K_M`; ≥ 40 GB (A100) → `qwen3.8:27b-q4_K_M` (Qwen3.8 nhỏ nhất là 27B, không vừa T4).
2. Mac: `uv run studio-web` → mở http://127.0.0.1:8765 → **⚙︎ Cài đặt** → dán 2 link.
3. Kéo thả ảnh vào trang chủ → Qwen tự nhận diện (20–90 s). Nhận sai thì bấm "Đây là mind map / nhân vật".
4. Mind map: bấm gợi ý của AI để áp dụng, hoặc gõ yêu cầu ("thêm 3 nhánh con cho …"); nút *Sinh icon minh hoạ*, *Tạo nhân vật từ sơ đồ*.
5. Nhân vật: chọn phong cách → *Tạo nhân vật* (mặc định nền trong suốt, PNG). Bấm chọn ảnh ưng ý → gợi ý biến thể, nhắn yêu cầu tiếng Việt, hoặc *Bộ biểu cảm tự động* (chọn cảm xúc, chân dung/toàn thân) → tải cả bộ .zip.

Cả hai cách (web + Qwen, Claude Desktop + MCP) dùng chung SQLite và `storage/`.

## MCP tools

| Tool | Việc làm |
|---|---|
| `studio_status`, `set_comfy_url` | Kiểm tra / đặt link Colab (link đổi mỗi phiên) |
| `list_sketches`, `open_sketch` | Liệt kê `inbox/`; làm sạch ảnh (OpenCV), tạo project, trả ảnh cho Claude nhìn |
| `submit_analysis` | VLM Router: nhận `UniversalSketchAnalysisResponse` (schema §4 của BRD) |
| `get_mindmap` | Lấy sơ đồ mới nhất kể cả chỉnh sửa trên web |
| `generate_character` | Workflow `sketch_to_2d`: ControlNet nét vẽ + hires pass, 5 style, 3 mức fidelity |
| `generate_variation` | Biến thể img2img từ ảnh đã chọn (giữ nhận diện + tư thế) |
| `generate_mindmap_icons` | Icon 512px cho từng nhánh, gắn lên sơ đồ |
| `get_job` | Chờ & tải kết quả về `storage/`, trả ảnh thu nhỏ |
| `generate_expressions` | Bộ biểu cảm (12 cảm xúc có sẵn + tuỳ chọn), cùng seed + khoá gương mặt IP-Adapter, nền trong suốt |
| `remove_background` | Sticker PNG trong suốt (rembg `isnet-anime`, chạy trên Mac) — ảnh mới đã tự xoá nền |

## So với BRD: MVP làm gì / để sau

| BRD | MVP |
|---|---|
| Next.js + Excalidraw/xyflow | Một trang HTML + React Flow (xyflow) qua CDN, không cần build |
| FastAPI + Postgres + Redis + R2 | FastAPI + SQLite + thư mục `storage/` (§9 Local Edition) |
| VLM Cloud API | Claude Desktop qua MCP (không tốn API) |
| ComfyUI + controlnet_aux + IPAdapter + PuLID + StoryDiffusion + BiRefNet | **Chỉ node core** của ComfyUI + ControlNet; hint nét vẽ tạo bằng OpenCV trên Mac; tách nền bằng rembg trên Mac. Không cần cài custom node trên Colab → ít lỗi |
| Workflow 2/3 (giữ mặt bằng IPAdapter/PuLID, sheet biểu cảm StoryDiffusion) | img2img + **IP-Adapter PLUS FACE** (khoá mặt từ ảnh cắt khuôn mặt) + cùng seed. Không dùng PuLID/InstantID vì chúng nhận diện mặt người thật (insightface), kém với mặt anime |
| Workflow 4 inpaint | Chưa làm |
| users / credits_ledger | Bỏ (một người dùng, local) |

## Test

```bash
uv run pytest -q
```

Test chạy toàn bộ pipeline với một ComfyUI giả lập ([tests/mock_comfy.py](../tests/mock_comfy.py)) kiểm tra workflow theo đúng tên input/output của các node core trong `ComfyUI/nodes.py`. Toàn bộ luồng cũng đã chạy thật trên Colab T4 (số đo trong README tiếng Anh).

## Lưu ý rủi ro

- **Khoá gương mặt** dùng custom node `cubiq/ComfyUI_IPAdapter_plus` (chế độ "maintenance only" từ 4/2025). Nếu bản ComfyUI mới làm hỏng node: chế độ *tự động* sẽ bỏ khoá mặt; bật khoá mặt thủ công sẽ báo lỗi rõ ràng. Khoá quá mạnh (>0.85) làm biểu cảm khó đổi vì IP-Adapter copy cả nét mặt của ảnh gốc.

- **Điều khoản Colab:** Colab miễn phí từng hạn chế/cảnh báo các UI Stable Diffusion và tunnel ra ngoài. Nếu bị ngắt hoặc hiện cảnh báo "disallowed code", dùng Colab Pro / Pay-as-you-go hoặc chuyển ComfyUI sang RunPod/Modal (chỉ cần đổi `COMFY_URL`).
- **Bảo mật tunnel:** link trycloudflare công khai, ComfyUI không có mật khẩu. Không chia sẻ link, tắt runtime khi xong.
- **Ảnh đính kèm trong chat không tới được MCP server**: luôn để ảnh trong `inbox/` (hoặc đưa đường dẫn tuyệt đối) và nhờ Claude `open_sketch` — Claude sẽ nhìn ảnh qua tool, server giữ file gốc cho ControlNet.
- Claude đọc chữ viết tay tốt nhưng không đảm bảo 100% như BRD ghi; sửa trực tiếp trên web studio.
- Hạn mức tin nhắn của Claude Pro: mỗi ảnh trả về tốn context; `get_job` chỉ trả ảnh thu nhỏ 384px.
