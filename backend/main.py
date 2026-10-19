"""
Main FastAPI Application Entrypoint.
Initializes the application, middleware, lifecycle events, and routes.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.db.session import init_db
from backend.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management.
    Handles startup database migration and clean shutdown.
    """
    # Startup: Ensure database schema is ready
    init_db()
    yield
    # Shutdown logic (if any)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade, research-informed biometric attendance analytics platform. "
        "Provides modern Deep Learning face recognition (YOLOv8 + FaceNet/ArcFace), "
        "anti-spoofing liveness detection (EAR), and classical Haar+LBPH baseline benchmarking."
    ),
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan
)

# Configure CORS Middleware for Streamlit and external frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Root"])
def root():
    """Root entry point providing API overview and navigation links."""
    return {
        "title": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": f"{settings.API_V1_PREFIX}/docs",
        "health_check": f"{settings.API_V1_PREFIX}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
