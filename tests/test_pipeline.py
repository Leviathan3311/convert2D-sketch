import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sketch_studio import comfy, config, db, service, web
from sketch_studio.schemas import UniversalSketchAnalysisResponse

GRAPH = {
    "title_vi": "Kế hoạch dự án",
    "nodes": [
        {"node_id": "n1", "shape_type": "circle", "text_content_vi": "Mục tiêu chính", "x_norm": 500, "y_norm": 200,
         "suggested_color_hex": "#FFADAD", "icon_prompt_en": "golden trophy"},
        {"node_id": "n2", "shape_type": "triangle", "text_content_vi": "Ý tưởng A \"mới\"", "x_norm": 250, "y_norm": 700,
         "suggested_color_hex": "#CAFFBF", "icon_prompt_en": "light bulb"},
        {"node_id": "n3", "shape_type": "diamond", "text_content_vi": "Quyết định", "x_norm": 750, "y_norm": 700,
         "suggested_color_hex": "#9BF6FF"},
    ],
    "edges": [
        {"edge_id": "e1", "source_node_id": "n1", "target_node_id": "n2", "arrow_type": "directed", "relation_label_vi": "bước 1"},
        {"edge_id": "e2", "source_node_id": "n1", "target_node_id": "n3", "arrow_type": "dashed"},
    ],
    "ai_expansion_suggestions_vi": ["Thêm nhánh ngân sách", "Tách ý tưởng A thành 2 nhánh"],
}

PROFILE = {
    "character_id": "astro_kid",
    "character_archetype": "young space explorer",
    "sketch_complexity_score": 4,
    "recommended_fidelity_mode": "balanced",
    "immutable_traits_en": ["cracked astronaut helmet", "short spiky hair"],
    "mutable_outfit_en": "torn orange cape over a white spacesuit",
    "color_palette": {"hair_color": "black", "eye_color": "blue", "primary_outfit_color": "white", "accent_color": "orange"},
    "contextual_suggestions": [
        {"label_vi": "Cười lớn", "category": "expression", "prompt_delta_en": "laughing, happy", "recommended_workflow": "expression_sheet"}
    ],
}


def make_sketch(name: str, w=900, h=1200) -> str:
    """Synthetic paper photo: uneven yellowish lighting + a dark shadow band + pencil shapes."""
    yy, xx = np.mgrid[0:h, 0:w]
    paper = (200 + 40 * xx / w - 50 * (yy > h * 0.7)).astype(np.uint8)
    img = cv2.merge([paper - 25, paper - 5, paper])
    cv2.circle(img, (w // 2, h // 4), 120, (60, 60, 60), 4)
    cv2.rectangle(img, (150, 700), (420, 900), (60, 60, 60), 4)
    cv2.line(img, (w // 2, h // 4 + 120), (285, 700), (60, 60, 60), 3)
    cv2.imwrite(str(config.INBOX / name), img)
    return name


def test_preprocess_removes_shadow_and_builds_hint():
    r = service.open_sketch(make_sketch("shadow.jpg"))
    p = db.get_project(r["project_id"])
    assert p["aspect"] == "portrait"
    clean = cv2.imread(str(config.STORAGE / p["sketch_clean"]), cv2.IMREAD_GRAYSCALE)
    # paper (lit and shadowed areas) should both become near-white
    assert clean[100, 50] > 230 and clean[1150, 50] > 230
    hint = cv2.imread(str(config.STORAGE / p["sketch_control"]), cv2.IMREAD_GRAYSCALE)
    assert hint.shape == (768, 512)
    assert (hint > 128).mean() < 0.1  # mostly black background, white lines
    assert (hint > 128).sum() > 500


def test_mindmap_flow_and_web_edit():
    pid = service.open_sketch(make_sketch("mm.jpg", 1200, 800))["project_id"]
    analysis = UniversalSketchAnalysisResponse.model_validate(
        {"detected_input_category": "mindmap_or_flowchart_diagram", "mindmap_graph": GRAPH}
    )
    r = service.submit_analysis(pid, analysis)
    assert r["mindmap_version"] == 1
    assert 'n2[/"Ý tưởng A #quot;mới#quot;"\\]' in r["mermaid_code"]
    assert 'n1 -->|"bước 1"| n2' in r["mermaid_code"]

    client = TestClient(web.app)
    data = client.get(f"/api/projects/{pid}").json()
    assert data["project"]["project_type"] == "mindmap_2d"
    graph = data["graph"]
    graph["nodes"][2]["text_content_vi"] = "Quyết định cuối"
    assert client.put(f"/api/projects/{pid}/mindmap", json=graph).json() == {"version": 2}
    assert "Quyết định cuối" in db.latest_graph(pid)[1]["mermaid_code"]

    graph["edges"].append({"edge_id": "bad", "source_node_id": "n1", "target_node_id": "ghost", "arrow_type": "directed"})
    assert client.put(f"/api/projects/{pid}/mindmap", json=graph).status_code == 422
    assert client.get("/").status_code == 200


def test_workflows_have_valid_links():
    from mock_comfy import validate, app

    app.state.uploads["hint.png"] = b""
    for wf in [
        comfy.character_workflow("p", (512, 768), 4, 1, "hint.png", "balanced"),
        comfy.character_workflow("p", (512, 768), 2, 1, None, "balanced", hires=False),
        comfy.variation_workflow("p", "hint.png", 0.5, 2, 1, "hint.png"),
        comfy.icons_workflow({"n1": "trophy", "n2": "bulb"}, 1),
    ]:
        assert validate(comfy.numbered(wf)[0]) == []


@pytest.mark.asyncio
async def test_character_end_to_end(mock_comfy):
    pid = service.open_sketch(make_sketch("char.jpg"))["project_id"]
    service.submit_analysis(pid, UniversalSketchAnalysisResponse.model_validate(
        {"detected_input_category": "character_or_object_sketch", "character_profile": PROFILE}))

    job = await service.generate_character(pid, style="chibi", batch_size=3, seed=42)
    assert "cracked astronaut helmet" in job["prompt"] and "chibi" in job["prompt"]
    first = await service.poll_job(job["job_id"], wait_seconds=0)
    assert first["status"] == "running"
    done = await service.poll_job(job["job_id"], wait_seconds=10)
    assert done["status"] == "done" and len(done["assets"]) == 3
    assert db.get_job(job["job_id"])["gpu_seconds"] == 8.0
    # polling again must not duplicate assets
    again = await service.poll_job(job["job_id"])
    assert len(again["assets"]) == 3

    wf = mock_comfy.state.prompts[-1]
    cn = next(n for n in wf.values() if n["class_type"] == "ControlNetApplyAdvanced")
    assert cn["inputs"]["strength"] == 0.8

    var = await service.generate_variation(done["assets"][0]["id"], "laughing, happy", batch_size=2)
    var_done = await service.poll_job(var["job_id"], wait_seconds=10)
    assert len(var_done["assets"]) == 2
    assert var_done["assets"][0]["parent_asset_id"] == done["assets"][0]["id"]


@pytest.mark.asyncio
async def test_concept_mindmap_uses_txt2img_and_icons(mock_comfy):
    pid = service.open_sketch(make_sketch("concept.jpg", 1200, 800))["project_id"]
    service.submit_analysis(pid, UniversalSketchAnalysisResponse.model_validate(
        {"detected_input_category": "character_concept_mindmap", "mindmap_graph": GRAPH, "character_profile": PROFILE}))
    job = await service.generate_character(pid, batch_size=1, hires=False)
    wf = mock_comfy.state.prompts[-1]
    assert not any(n["class_type"] == "ControlNetApplyAdvanced" for n in wf.values())
    assert (await service.poll_job(job["job_id"], 10))["status"] == "done"

    icons = await service.generate_icons(pid)
    done = await service.poll_job(icons["job_id"], 10)
    assert {a["type"] for a in done["assets"]} == {"icon"} and len(done["assets"]) == 2
    nodes = {n["node_id"]: n for n in db.latest_graph(pid)[1]["nodes"]}
    assert nodes["n1"]["icon_url"].startswith("/storage/") and nodes["n3"]["icon_url"] is None
    assert db.get_project(pid)["project_type"] == "hybrid_sketchnote"


@pytest.mark.asyncio
async def test_comfy_unreachable_message():
    db.set_setting("comfy_url", "http://127.0.0.1:9")
    try:
        with pytest.raises(comfy.ComfyError, match="Colab"):
            await service.client().status()
    finally:
        db.set_setting("comfy_url", "")


@pytest.fixture
def web_llm(mock_comfy):
    db.set_setting("comfy_url", mock_comfy.state.url)
    db.set_setting("llm_url", mock_comfy.state.url)
    mock_comfy.state.chat_calls.clear()
    with TestClient(web.app) as c:
        yield c, mock_comfy


def _wait_analyzed(client, pid):
    import time
    for _ in range(100):
        p = client.get(f"/api/projects/{pid}").json()
        if p["project"]["status"] != "analyzing":
            return p
        time.sleep(0.05)
    raise AssertionError("analysis did not finish")


def test_web_upload_analyze_mindmap_and_edit(web_llm):
    client, mock = web_llm
    mock.state.category = "mindmap_or_flowchart_diagram"
    name = make_sketch("web_mm.jpg", 1200, 800)
    with open(config.INBOX / name, "rb") as f:
        pid = client.post("/api/upload", files={"file": ("so do.jpg", f, "image/jpeg")}).json()["project_id"]
    p = _wait_analyzed(client, pid)
    assert p["project"]["status"] == "analyzed", p["project"]["status_detail"]
    assert p["graph"]["title_vi"] == "Kế hoạch dự án"
    texts = {n["node_id"]: n["text_content_vi"] for n in p["graph"]["nodes"]}
    assert texts["n3"] == "Quyết định"  # proofreading fixed the diacritics
    assert texts["n2"] == GRAPH["nodes"][1]["text_content_vi"]  # but could not rewrite a word
    # first call classifies with the image attached, second extracts the graph
    assert mock.state.chat_calls[0]["messages"][1]["images"]
    assert "icon_url" not in json.dumps(mock.state.chat_calls[1]["format"])

    r = client.post(f"/api/projects/{pid}/mindmap/edit", json={"instruction_vi": "thêm nhánh ngân sách"})
    assert r.json() == {"version": 2}
    nodes = db.latest_graph(pid)[1]["nodes"]
    assert any(n["text_content_vi"] == "Ngân sách" for n in nodes)

    job = client.post(f"/api/projects/{pid}/icons").json()
    assert job["kind"] == "icons"


def test_web_character_flow_with_retry(web_llm):
    client, mock = web_llm
    mock.state.category = "character_or_object_sketch"
    mock.state.fail_once = True  # classifier returns bad JSON once -> retried
    name = make_sketch("web_char.jpg")
    with open(config.INBOX / name, "rb") as f:
        pid = client.post("/api/upload", files={"file": (name, f, "image/jpeg")}).json()["project_id"]
    p = _wait_analyzed(client, pid)
    assert p["project"]["status"] == "analyzed", p["project"]["status_detail"]
    assert p["profile"]["character_id"] == "astro_kid"

    job = client.post(f"/api/projects/{pid}/generate", json={"style": "chibi", "batch_size": 2, "extra_vi": "đang cười"}).json()
    assert "laughing, wool hat" in job["prompt"]
    for _ in range(3):
        client.post(f"/api/jobs/{job['job_id']}/refresh")
    assets = client.get(f"/api/projects/{pid}").json()["assets"]
    assert len(assets) == 2

    chip = client.post(f"/api/assets/{assets[0]['id']}/variation", json={"prompt_delta_en": "angry", "strength": 0.4}).json()
    assert "angry" in chip["prompt"]
    asked = client.post(f"/api/assets/{assets[0]['id']}/variation", json={"request_vi": "đội mũ len"}).json()
    assert "wool hat" in asked["prompt"]

    # a coloured object gets its own BREAK chunk + cross-bleed negatives (red balloon came out pink on SD1.5)
    red = client.post(f"/api/assets/{assets[0]['id']}/variation", json={"request_vi": "cầm bóng bay màu đỏ"}).json()
    assert red["prompt"].endswith("BREAK (red balloon:1.35)") and "holding a red balloon" in red["prompt"]
    wf = mock.state.prompts[-1]
    assert any(n["class_type"] == "ConditioningConcat" for n in wf.values())
    neg_texts = [n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert any("red dress" in t for t in neg_texts)


def test_llm_unreachable_is_reported(web_llm):
    client, _ = web_llm
    db.set_setting("llm_url", "http://127.0.0.1:9")
    pid = service.open_sketch(make_sketch("down.jpg"))["project_id"]
    client.post(f"/api/projects/{pid}/analyze", json={})
    p = _wait_analyzed(client, pid)
    assert p["project"]["status"] == "error" and "Colab" in p["project"]["status_detail"]
    settings = client.get("/api/settings").json()
    assert "error" in settings["llm"] and "models" not in settings["llm"]


@pytest.mark.asyncio
async def test_transparent_character_and_expression_pack(mock_comfy):
    import io
    from PIL import Image

    db.set_setting("comfy_url", mock_comfy.state.url)
    pid = service.open_sketch(make_sketch("expr.jpg"))["project_id"]
    service.submit_analysis(pid, UniversalSketchAnalysisResponse.model_validate(
        {"detected_input_category": "character_or_object_sketch", "character_profile": PROFILE}))
    job = await service.generate_character(pid, style="chibi", batch_size=1, hires=False)
    asset = (await service.poll_job(job["job_id"], 10))["assets"][0]
    assert Image.open(config.STORAGE / asset["path"]).mode == "RGBA"  # background removed

    r = await service.generate_expressions(asset["id"], ["happy", "angry", "cry"], [("say mê", "infatuated, heart eyes")])
    assert r["emotions"] == {"happy": "Vui vẻ", "angry": "Tức giận", "cry": "Khóc", "custom1": "say mê"}
    wf = mock_comfy.state.prompts[-1]
    seeds = {n["inputs"]["seed"] for n in wf.values() if n["class_type"] == "KSampler"}
    assert len(seeds) == 1  # same seed for every emotion -> consistent identity
    texts = [n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert any(t.startswith("(angry, glaring eyes") and ":1.3)" in t and "close-up portrait" in t and "chibi" in t for t in texts)
    src = next(n for n in wf.values() if n["class_type"] == "LoadImage")["inputs"]["image"]
    up = Image.open(io.BytesIO(mock_comfy.state.uploads[src]))
    assert up.size == (640, 640) and up.mode == "RGB"  # bust crop, flattened onto white

    done = await service.poll_job(r["job_id"], 10)
    labels = sorted(json.loads(a["meta_json"])["label_vi"] for a in done["assets"])
    assert labels == ["Khóc", "Tức giận", "Vui vẻ", "say mê"]
    assert all(a["type"] == "expression" and a["parent_asset_id"] == asset["id"] for a in done["assets"])

    with TestClient(web.app) as client:
        z = client.get(f"/api/jobs/{r['job_id']}/zip")
        assert z.status_code == 200 and z.content[:2] == b"PK"
        assert len(client.get("/api/expressions").json()) == 12
    with pytest.raises(ValueError, match="không có trong danh sách"):
        await service.generate_expressions(asset["id"], ["dance"])


@pytest.mark.asyncio
async def test_face_lock_auto_on_and_off(mock_comfy):
    import io
    from PIL import Image

    db.set_setting("comfy_url", mock_comfy.state.url)
    pid = service.open_sketch(make_sketch("face.jpg"))["project_id"]
    service.submit_analysis(pid, UniversalSketchAnalysisResponse.model_validate(
        {"detected_input_category": "character_or_object_sketch", "character_profile": PROFILE}))
    job = await service.generate_character(pid, batch_size=1, hires=False, transparent=False)
    asset = (await service.poll_job(job["job_id"], 10))["assets"][0]

    # auto: Colab has IP-Adapter -> expressions get PLUS FACE from a 512px face crop
    r = await service.generate_expressions(asset["id"], ["happy", "sad"])
    assert r["face_lock"] == 0.65
    wf = mock_comfy.state.prompts[-1]
    by_cls = {n["class_type"]: n for n in wf.values()}
    assert by_cls["IPAdapterUnifiedLoader"]["inputs"]["preset"] == "PLUS FACE (portraits)"
    ipa = by_cls["IPAdapterAdvanced"]["inputs"]
    assert ipa["weight"] == 0.65 and ipa["end_at"] == 0.6
    ipa_id = next(k for k, n in wf.items() if n["class_type"] == "IPAdapterAdvanced")
    assert all(n["inputs"]["model"] == [ipa_id, 0] for n in wf.values() if n["class_type"] == "KSampler")
    ref = Image.open(io.BytesIO(mock_comfy.state.uploads[wf[ipa["image"][0]]["inputs"]["image"]]))
    assert ref.size == (512, 512)
    assert (await service.poll_job(r["job_id"], 10))["status"] == "done"

    v = await service.generate_variation(asset["id"], "angry", face_lock=0.9)
    assert v["face_lock"] == 0.9
    assert any(n["class_type"] == "IPAdapterAdvanced" and n["inputs"]["weight"] == 0.9
               for n in mock_comfy.state.prompts[-1].values())

    # explicit off, and auto when the custom node is missing on Colab
    assert (await service.generate_variation(asset["id"], "angry", face_lock=0))["face_lock"] == 0.0
    mock_comfy.state.ipadapter = False
    try:
        r = await service.generate_expressions(asset["id"], ["happy"])
        assert r["face_lock"] == 0.0
        assert not any(n["class_type"].startswith("IPAdapter") for n in mock_comfy.state.prompts[-1].values())
        assert (await service.client().status())["face_lock"] is False
        with pytest.raises(ValueError, match="IP-Adapter"):
            await service.generate_variation(asset["id"], "angry", face_lock=0.7)
    finally:
        mock_comfy.state.ipadapter = True


def test_merge_revision_restores_what_small_models_drop():
    from sketch_studio import llm
    from sketch_studio.schemas import MindMapGraphStructure

    old = json.loads(json.dumps(GRAPH))
    old["nodes"][0]["icon_url"] = "/storage/x.png"
    new = json.loads(json.dumps(GRAPH))
    for n in new["nodes"]:
        n["x_norm"], n["y_norm"] = 370, 450  # model moved everything onto one spot
    new["edges"][0]["relation_label_vi"] = None  # and dropped a label
    new["nodes"] += [{**GRAPH["nodes"][2], "node_id": f"n{i}", "text_content_vi": f"Mới {i}", "x_norm": 370, "y_norm": 450}
                     for i in (4, 5, 6)]
    new["edges"] += [{"edge_id": f"e{i}", "source_node_id": "n2", "target_node_id": f"n{i}", "arrow_type": "directed"}
                     for i in (4, 5, 6)]
    merged = llm.merge_revision(old, MindMapGraphStructure.model_validate(new))
    by_id = {n.node_id: n for n in merged.nodes}
    assert (by_id["n1"].x_norm, by_id["n1"].y_norm, by_id["n1"].icon_url) == (500, 200, "/storage/x.png")
    assert merged.edges[0].relation_label_vi == "bước 1"
    boxes = [(n.x_norm, n.y_norm) for n in merged.nodes]
    for i, (x, y) in enumerate(boxes):  # no two nodes overlap
        assert all(abs(x - x2) >= 125 or abs(y - y2) >= 95 for x2, y2 in boxes[i + 1:])
    assert all(0 <= v <= 1000 for x, y in boxes for v in (x, y))


def test_expression_prompts_drop_conflicting_traits():
    profile = {**PROFILE, "character_archetype": "cheerful little girl",
               "immutable_traits_en": ["round head with big dot eyes and simple smile", "short bangs", "smiling face"]}
    sad = comfy.character_prompt(profile, "chibi", suffix=comfy.BUST_SUFFIX, focus=comfy.EXPRESSIONS["sad"][1])
    assert sad.startswith("(sad,") and "smile" not in sad and "cheerful" not in sad and ", face," not in sad
    assert "(black hair, blue eyes:1.15)" in sad
    assert "eyebrow" not in " ".join(p for _, p in comfy.EXPRESSIONS.values())
    plain = comfy.character_prompt(profile, "anime")  # a normal generation keeps the DNA as written
    assert "simple smile" in plain and not plain.startswith("(")
    wf = comfy.expressions_workflow({"sad": sad}, "src.png", 0.6, 1)
    assert "unibrow" in wf["neg"]["inputs"]["text"]
