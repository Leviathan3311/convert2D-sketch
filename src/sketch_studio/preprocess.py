"""Paper-photo cleanup (BRD §2.1 step 1): remove shadows/paper tint, build a ControlNet hint."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

# Generation sizes (SD1.5-friendly, multiples of 64) keyed by aspect.
SIZES = {"portrait": (512, 768), "landscape": (768, 512), "square": (640, 640)}


def load_rgb(path: Path, max_side: int = 1280) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def pick_aspect(w: int, h: int) -> str:
    r = w / h
    return "landscape" if r > 1.2 else "portrait" if r < 0.83 else "square"


def clean_sketch(gray: np.ndarray) -> np.ndarray:
    """Divide out the blurred paper background so shadows and yellow tint disappear."""
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    norm = cv2.divide(gray, bg, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def control_hint(clean: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """White lines on black, letterboxed to the generation size (ComfyUI center-crops otherwise)."""
    _, lines = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.dilate(lines, np.ones((2, 2), np.uint8))
    tw, th = size
    h, w = lines.shape
    s = min(tw / w, th / h) * 0.94
    resized = cv2.resize(lines, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw), np.uint8)
    y, x = (th - resized.shape[0]) // 2, (tw - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def process(src: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = load_rgb(src)
    raw_path = out_dir / "sketch_raw.jpg"
    img.save(raw_path, quality=90)

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    clean = clean_sketch(gray)
    clean_path = out_dir / "sketch_clean.png"
    cv2.imwrite(str(clean_path), clean)

    aspect = pick_aspect(img.width, img.height)
    control_path = out_dir / "sketch_control.png"
    cv2.imwrite(str(control_path), control_hint(clean, SIZES[aspect]))
    return {"raw": raw_path, "clean": clean_path, "control": control_path, "aspect": aspect}


def thumbnail_jpeg(path: Path, max_side: int = 1024, quality: int = 85) -> bytes:
    """Downscaled JPEG for returning to Claude (keeps tool results small)."""
    img = Image.open(path)
    img = img.convert("RGBA") if img.mode in ("P", "LA") else img
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[3])
        img = bg
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _on_white(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, "white")
    return Image.alpha_composite(bg, img).convert("RGB")


def bust_box(img: Image.Image) -> tuple[int, int, int, int]:
    """Square head-and-shoulders box from the character's bounding box (alpha if cut out, else non-white)."""
    if img.mode == "RGBA":
        mask = np.array(img.split()[3]) > 20
    else:
        mask = np.array(img.convert("L")) < 235
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0, 0, img.width, img.height)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    side = int(min(max(w * 0.8, h * 0.42), img.width, img.height))
    cx = (x0 + x1) // 2
    left = int(min(max(cx - side // 2, 0), img.width - side))
    top = int(min(max(y0 - side * 0.05, 0), img.height - side))
    return (left, top, left + side, top + side)


def face_box(img: Image.Image) -> tuple[int, int, int, int]:
    """Square around the head (IP-Adapter face reference): centre of the head-and-shoulders box,
    nudged down so the chin is included."""
    l, t, r, b = bust_box(img)
    side = int((r - l) * 0.75)
    left = l + ((r - l) - side) // 2
    top = min(t + int((r - l) * 0.08), b - side)
    return (left, top, left + side, top + side)


def flat_png(path: Path, bust: bool = False, size: int = 640, face: bool = False) -> bytes:
    """PNG for img2img: transparent cut-outs flattened onto white (ComfyUI drops alpha -> black otherwise),
    optionally cropped to a head-and-shoulders square, or to the face for IP-Adapter."""
    img = Image.open(path)
    if bust or face:
        # locate the character on its alpha; opaque images (busy backgrounds) get a cut-out first
        img = img if img.mode == "RGBA" else Image.open(BytesIO(cutout(path.read_bytes())))
        box = face_box(img) if face else bust_box(img)
        img = img.crop(box).resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    _on_white(img).save(buf, "PNG")
    return buf.getvalue()


_rembg_session = None


def cutout(data: bytes) -> bytes:
    """Transparent PNG via rembg isnet-anime (CPU, ~1-3 s/image on M1). Session is loaded once."""
    global _rembg_session
    from rembg import new_session, remove

    if _rembg_session is None:
        _rembg_session = new_session("isnet-anime")
    return remove(data, session=_rembg_session)
