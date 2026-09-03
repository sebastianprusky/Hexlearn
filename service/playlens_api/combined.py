from __future__ import annotations

from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .app import create_app
from .local_game import inject_playlens_loader


GAME_DIR = Path(__file__).resolve().parents[2] / "game" / "hextris"
EXTENSION_DIR = Path(__file__).resolve().parents[2] / "extension"
if not (GAME_DIR / "index.html").exists():
    raise RuntimeError(
        "Clean Hextris copy is missing. Follow playlens/game/README.md before starting Hexlearn."
    )

app = create_app()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
def local_game() -> HTMLResponse:
    html = inject_playlens_loader((GAME_DIR / "index.html").read_text(encoding="utf-8"))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/playlens/core.js", include_in_schema=False)
def playlens_core() -> FileResponse:
    return FileResponse(
        EXTENSION_DIR / "core.js", media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/playlens/content.js", include_in_schema=False)
def playlens_content() -> FileResponse:
    return FileResponse(
        EXTENSION_DIR / "content.js", media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/playlens/local-bootstrap.js", include_in_schema=False)
def playlens_local_bootstrap() -> FileResponse:
    return FileResponse(
        EXTENSION_DIR / "local-bootstrap.js", media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


app.mount("/", StaticFiles(directory=GAME_DIR, html=True), name="hextris")
