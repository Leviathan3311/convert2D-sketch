"""SQLite store (BRD §9: SQLite + local folder instead of Postgres/S3 for the local edition)."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, title TEXT, project_type TEXT, input_category TEXT,
    sketch_raw TEXT, sketch_clean TEXT, sketch_control TEXT, aspect TEXT,
    status TEXT, created_at TEXT, status_detail TEXT
);
CREATE TABLE IF NOT EXISTS mindmap_graphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, version INTEGER,
    graph_json TEXT, mermaid_source TEXT, source TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS character_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, dna_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, project_id TEXT, job_id TEXT, parent_asset_id TEXT, type TEXT,
    path TEXT, seed INTEGER, meta_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY, project_id TEXT, kind TEXT, engine_type TEXT, comfy_prompt_id TEXT,
    status TEXT, params_json TEXT, error TEXT, gpu_seconds REAL, created_at TEXT, finished_at TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(projects)")}
        if "status_detail" not in cols:  # added with the web-only (Ollama) flow
            c.execute("ALTER TABLE projects ADD COLUMN status_detail TEXT")


def get_setting(key: str, default: str = "") -> str:
    with conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))


def create_project(pid: str, title: str, raw: str, clean: str, control: str, aspect: str) -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (pid, title, None, None, raw, clean, control, aspect, "uploaded", now(), None),
        )


def update_project(pid: str, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE projects SET {sets} WHERE id=?", (*fields.values(), pid))


def get_project(pid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def list_projects() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM projects ORDER BY created_at DESC")]


def save_graph(pid: str, graph: dict, source: str) -> int:
    with conn() as c:
        version = c.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM mindmap_graphs WHERE project_id=?", (pid,)
        ).fetchone()[0]
        c.execute(
            "INSERT INTO mindmap_graphs (project_id, version, graph_json, mermaid_source, source, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (pid, version, json.dumps(graph, ensure_ascii=False), graph.get("mermaid_code", ""), source, now()),
        )
    return version


def latest_graph(pid: str) -> tuple[int, dict] | None:
    with conn() as c:
        row = c.execute(
            "SELECT version, graph_json FROM mindmap_graphs WHERE project_id=? ORDER BY version DESC LIMIT 1", (pid,)
        ).fetchone()
    return (row["version"], json.loads(row["graph_json"])) if row else None


def save_profile(pid: str, dna: dict) -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO character_profiles (project_id, dna_json, created_at) VALUES (?,?,?)",
            (pid, json.dumps(dna, ensure_ascii=False), now()),
        )


def latest_profile(pid: str) -> dict | None:
    with conn() as c:
        row = c.execute(
            "SELECT dna_json FROM character_profiles WHERE project_id=? ORDER BY id DESC LIMIT 1", (pid,)
        ).fetchone()
    return json.loads(row["dna_json"]) if row else None


def create_job(pid: str, kind: str, engine_type: str, comfy_prompt_id: str, params: dict) -> str:
    jid = new_id("job")
    with conn() as c:
        c.execute(
            "INSERT INTO generation_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (jid, pid, kind, engine_type, comfy_prompt_id, "running", json.dumps(params), None, None, now(), None),
        )
    return jid


def get_job(jid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM generation_jobs WHERE id=?", (jid,)).fetchone()
    if not row:
        return None
    job = dict(row)
    job["params"] = json.loads(job.pop("params_json"))
    return job


def claim_job(jid: str) -> bool:
    """Atomically move running -> collecting so MCP and web never download the same results twice."""
    with conn() as c:
        cur = c.execute("UPDATE generation_jobs SET status='collecting' WHERE id=? AND status='running'", (jid,))
    return cur.rowcount == 1


def finish_job(jid: str, status: str, error: str | None = None) -> None:
    with conn() as c:
        c.execute(
            "UPDATE generation_jobs SET status=?, error=?, finished_at=? WHERE id=?", (status, error, now(), jid)
        )


def list_jobs(pid: str) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT id, kind, status, error, created_at, finished_at FROM generation_jobs"
            " WHERE project_id=? ORDER BY created_at DESC",
            (pid,),
        )
        return [dict(r) for r in rows]


def add_asset(pid: str, job_id: str | None, parent: str | None, type_: str, path: str, seed: int | None, meta: dict) -> str:
    aid = new_id("ast")
    with conn() as c:
        c.execute(
            "INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, pid, job_id, parent, type_, path, seed, json.dumps(meta, ensure_ascii=False), now()),
        )
    return aid


def get_asset(aid: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM assets WHERE id=?", (aid,)).fetchone()
    return dict(row) if row else None


def list_assets(pid: str, job_id: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM assets WHERE project_id=?", [pid]
    if job_id:
        q, args = q + " AND job_id=?", args + [job_id]
    with conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY created_at", args)]
