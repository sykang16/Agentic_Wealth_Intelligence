"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.src.api.routes import chat, plaid, portfolio, profiling, recommendations, sessions, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load data on startup."""
    from backend.src.api.dependencies import get_aggregator, get_log_repository
    from backend.src.common.tracing import configure_tracing

    configure_tracing()

    try:
        get_aggregator()
    except Exception:
        pass  # Data may not exist yet; endpoints will fail gracefully

    # Ensure DB tables are created on startup
    try:
        get_log_repository()
    except Exception:
        pass

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic Wealth Intelligence API",
        description="AI-powered wealth management API with multi-agent orchestration.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(chat.router)
    app.include_router(portfolio.router)
    app.include_router(profiling.router)
    app.include_router(recommendations.router)
    app.include_router(sessions.router)
    app.include_router(plaid.router)
    app.include_router(users.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


# Module-level app instance for uvicorn
app = create_app()
