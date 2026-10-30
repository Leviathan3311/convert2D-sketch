"""MCP bridge (BRD §9–10): Claude Desktop (Claude Pro) is the VLM/LLM; these tools are its hands."""

import functools
import inspect
import json
import webbrowser

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import config, db, preprocess, service, web
from .comfy import STYLE_PRESETS
from .schemas import UniversalSketchAnalysisResponse

INSTRUCTIONS = f"""Sketch-to-2D Studio. Quy trình:
1. list_sketches → open_sketch(file): nhận ảnh phác thảo + project_id. TỰ BẠN là VLM: nhìn ảnh và phân loại.
2. submit_analysis(project_id, analysis) theo đúng schema UniversalSketchAnalysisResponse:
   - mindmap_or_flowchart_diagram → mindmap_graph (đọc chính xác chữ tiếng Việt, đúng shape_type, toạ độ 0-1000
     theo vị trí trên giấy, edges theo mũi tên). Web studio vẽ sơ đồ vector tại {config.WEB_URL}.
   - character_or_object_sketch → character_profile (traits/outfit/palette bằng tiếng Anh cho Stable Diffusion).
   - character_concept_mindmap → mindmap_graph + character_profile tổng hợp từ các nhánh.
3. Nhân vật: generate_character → get_job(job_id) (lặp lại tới khi status=done) → cho người dùng xem ảnh.
   Biến thể: generate_variation(asset_id, prompt_delta_en). Bộ biểu cảm: generate_expressions(asset_id).
   Ảnh nhân vật đã tự xoá nền (PNG trong suốt); remove_background chỉ cần cho ảnh cũ.
4. Mind map minh hoạ: thêm icon_prompt_en cho nhánh chính (submit_analysis lại) → generate_mindmap_icons → get_job.
Người dùng có thể sửa sơ đồ trên web; luôn gọi get_mindmap trước khi chỉnh tiếp để không ghi đè sửa đổi của họ.
ComfyUI chạy trên Colab qua link trycloudflare đổi mỗi phiên: dùng set_comfy_url khi người dùng dán link mới."""

mcp = MCPServer("sketch-studio", instructions=INSTRUCTIONS)
EXPECTED = (ValueError, FileNotFoundError, RuntimeError)  # ComfyError is a RuntimeError


def tool(fn):
    """Register a tool; expected failures become ToolError so their message reaches Claude
    (mcp 2.x hides the text of any other exception)."""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            try:
                return await fn(*a, **kw)
            except EXPECTED as e:
                raise ToolError(str(e)) from e
    else:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except EXPECTED as e:
                raise ToolError(str(e)) from e
    return mcp.tool(structured_output=False)(wrapper)


def _thumbs(assets: list[dict], max_side: int = 384) -> list:
    out = []
    for a in assets:
        out.append(f"asset_id={a['id']} type={a['type']} file={config.STORAGE / a['path']}")
        out.append(Image(data=preprocess.thumbnail_jpeg(config.STORAGE / a["path"], max_side), format="jpeg"))
    return out


@tool
async def studio_status() -> str:
    """Trạng thái hệ thống: COMFY_URL, GPU Colab, model còn thiếu, web studio, ảnh trong inbox."""
    comfy_url = db.get_setting("comfy_url", config.COMFY_URL)
    try:
        gpu = await service.client().status()
    except Exception as e:
        gpu = f"KHÔNG KẾT NỐI: {e}"
    return (
        f"web_studio: {config.WEB_URL}\ncomfy_url: {comfy_url or '(chưa đặt)'}\ncomfy: {gpu}\n"
        f"inbox: {config.INBOX} ({len(service.list_sketches())} ảnh)"
    )


@tool
async def set_comfy_url(url: str) -> str:
    """Lưu link ComfyUI (vd https://xxxx.trycloudflare.com do notebook Colab in ra) và kiểm tra kết nối."""
    db.set_setting("comfy_url", url.strip().rstrip("/"))
    try:
        return f"Đã lưu. ComfyUI OK: {await service.client().status()}"
    except Exception as e:
        return f"Đã lưu nhưng chưa kết nối được: {e}"


@tool
def list_sketches() -> str:
    """Liệt kê ảnh phác thảo trong thư mục inbox."""
    files = service.list_sketches()
    return f"inbox: {config.INBOX}\n" + ("\n".join(files) if files else "(trống — bỏ ảnh chụp giấy vào đây)")


@tool
def open_sketch(file: str) -> list:
    """Mở ảnh phác thảo (tên file trong inbox hoặc đường dẫn tuyệt đối): làm sạch nền giấy bằng OpenCV,
    tạo project, trả ảnh về để bạn phân tích rồi gọi submit_analysis."""
    r = service.open_sketch(file)
    return [
        f"project_id={r['project_id']} aspect={r['aspect']} web={r['url']}\n"
        "Hãy phân loại ảnh và gọi submit_analysis.",
        Image(data=preprocess.thumbnail_jpeg(r["raw_path"], 1280), format="jpeg"),
    ]


@tool
def submit_analysis(project_id: str, analysis: UniversalSketchAnalysisResponse, open_browser: bool = True) -> str:
    """VLM Router: lưu kết quả phân tích ảnh (mind map và/hoặc Character DNA). Gửi lại để tạo phiên bản mới."""
    r = service.submit_analysis(project_id, analysis)
    if open_browser:
        webbrowser.open(r["url"])
    lines = [f"Đã lưu ({r['category']}). Web studio: {r['url']}"]
    if "mindmap_version" in r:
        lines.append(f"mind map v{r['mindmap_version']}; mermaid:\n{r['mermaid_code']}")
    if analysis.character_profile:
        lines.append("Bước tiếp: generate_character(project_id, style=...) với style ∈ " + ", ".join(STYLE_PRESETS))
    return "\n".join(lines)


@tool
def get_mindmap(project_id: str) -> str:
    """Lấy phiên bản mind map mới nhất (gồm cả chỉnh sửa người dùng làm trên web) dạng JSON."""
    latest = db.latest_graph(project_id)
    if not latest:
        return "Project chưa có mind map"
    return f"version {latest[0]}\n" + json.dumps(latest[1], ensure_ascii=False, indent=1)


@tool
async def generate_character(
    project_id: str, style: str = "anime", batch_size: int = 4, seed: int | None = None,
    fidelity: str | None = None, extra_prompt_en: str = "", hires: bool = True,
) -> str:
    """Workflow sketch_to_2d: sinh nhân vật 2D trên ComfyUI/Colab từ Character DNA (+ ControlNet nét vẽ nếu là sketch).
    style: anime|chibi|flat_vector|cartoon|watercolor. fidelity: high_fidelity|balanced|creative_fix (mặc định theo DNA).
    Trả job_id ngay; gọi get_job để lấy ảnh (~30-90s trên T4)."""
    r = await service.generate_character(project_id, style, batch_size, seed, fidelity, extra_prompt_en, hires)
    return f"job_id={r['job_id']} seed={r['seed']}\nprompt: {r['prompt']}\nGọi get_job('{r['job_id']}')."


@tool
async def generate_variation(
    asset_id: str, prompt_delta_en: str, strength: float = 0.55, keep_pose: bool = True,
    batch_size: int = 2, seed: int | None = None, face_lock: float | None = None,
) -> str:
    """Biến thể từ một ảnh đã chọn (biểu cảm/trang phục/phụ kiện) giữ nhận diện bằng img2img.
    strength 0.35 (giữ sát) → 0.75 (đổi nhiều). keep_pose dùng lại nét sketch gốc.
    face_lock: khoá mặt IP-Adapter 0-1 (None = tự động 0.7 nếu Colab có IP-Adapter, 0 = tắt)."""
    r = await service.generate_variation(asset_id, prompt_delta_en, strength, keep_pose, batch_size, seed, face_lock)
    return f"job_id={r['job_id']} seed={r['seed']}\nprompt: {r['prompt']}\nGọi get_job('{r['job_id']}')."


@tool
async def generate_expressions(
    asset_id: str, emotions: list[str] | None = None, custom_en: list[str] | None = None,
    framing: str = "bust", strength: float | None = None, face_lock: float | None = None,
) -> str:
    """Bộ biểu cảm từ một ảnh nhân vật đã chọn (cùng seed, chỉ đổi cảm xúc), nền trong suốt.
    emotions: key trong [happy, laugh, sad, cry, angry, surprised, scared, shy, wink, sleepy, confident, confused] (mặc định 8 cái đầu). custom_en: cảm xúc khác, prompt tiếng Anh.
    framing: bust (chân dung, kiểu sticker) | full (toàn thân).
    face_lock: độ khoá mặt IP-Adapter 0-1 (None = tự động 0.65 nếu Colab có IP-Adapter, 0 = tắt)."""
    custom = [(c, c) for c in custom_en or []]
    r = await service.generate_expressions(asset_id, emotions, custom, framing, strength, face_lock=face_lock)
    return (f"job_id={r['job_id']} khoá mặt={r['face_lock']} cảm xúc: {', '.join(r['emotions'].values())}. "
            f"Gọi get_job('{r['job_id']}').")


@tool
async def generate_mindmap_icons(project_id: str, node_ids: list[str] | None = None) -> str:
    """Mind map minh hoạ: sinh icon 2D 512px cho các node có icon_prompt_en và gắn vào sơ đồ trên web."""
    r = await service.generate_icons(project_id, node_ids)
    return f"job_id={r['job_id']}. Gọi get_job('{r['job_id']}')."


@tool
async def get_job(job_id: str, wait_seconds: float = 40) -> list:
    """Chờ tối đa wait_seconds rồi trả trạng thái job; khi xong trả kèm ảnh thu nhỏ + asset_id."""
    job = await service.poll_job(job_id, min(wait_seconds, 50))
    head = f"job {job_id}: {job['status']}"
    if job["status"] == "error":
        return [f"{head}\n{job['error']}"]
    if job["status"] != "done":
        return [f"{head} — ComfyUI vẫn đang chạy, gọi get_job lại."]
    url = service.project_url(job["project_id"])
    return [f"{head}, {len(job['assets'])} ảnh. Xem đầy đủ: {url}", *_thumbs(job["assets"])]


@tool
def remove_background(asset_id: str) -> list:
    """Tách nền ảnh thành sticker PNG trong suốt (rembg isnet-anime, chạy trên Mac)."""
    r = service.remove_background(asset_id)
    return [f"sticker asset_id={r['asset_id']} file={r['path']}", Image(data=preprocess.thumbnail_jpeg(r["path"], 384), format="jpeg")]


def main() -> None:
    db.init()
    web.start_background()
    mcp.run("stdio")


if __name__ == "__main__":
    main()
