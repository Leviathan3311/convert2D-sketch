import os
import socket
import tempfile
import threading
import time

import pytest

# Isolate storage before the package reads config.
os.environ["STUDIO_HOME"] = tempfile.mkdtemp(prefix="studio_test_")
os.environ["STUDIO_WEB_PORT"] = "8799"

import uvicorn  # noqa: E402

from sketch_studio import db  # noqa: E402

db.init()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_comfy():
    from mock_comfy import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    url = app.state.url = f"http://127.0.0.1:{port}"
    db.set_setting("comfy_url", url)
    yield app
    server.should_exit = True
