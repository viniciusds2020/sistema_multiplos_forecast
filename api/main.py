"""FastAPI application - ForecastForge backend."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.dependencies import get_data
from api.routers import general, overview, explorer, similarity, forecasting, comparison


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load synthetic data on startup."""
    print("Gerando dados sinteticos...")
    get_data()
    print("Dados prontos.")
    yield


app = FastAPI(title="ForecastForge API", lifespan=lifespan)

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(general.router, prefix="/api")
app.include_router(overview.router, prefix="/api")
app.include_router(explorer.router, prefix="/api")
app.include_router(similarity.router, prefix="/api")
app.include_router(forecasting.router, prefix="/api")
app.include_router(comparison.router, prefix="/api")

# Serve React SPA build in production
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"


@app.get("/{path:path}")
async def serve_spa(request: Request, path: str):
    """Catch-all: serve static files from frontend/dist or fallback to index.html."""
    if _frontend_dist.exists():
        file_path = _frontend_dist / path
        if file_path.is_file():
            return FileResponse(file_path)
        index = _frontend_dist / "index.html"
        if index.exists():
            return FileResponse(index)
    return {"detail": "Frontend not built. Run: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
