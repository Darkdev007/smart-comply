import time
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from package.pipeline import screen_entity

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":     record.levelname,
            "message":   record.getMessage(),
        }
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger("smart_comply")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Smart Comply API starting up — model and watchlist index ready.")
    yield
    logger.info("Smart Comply API shutting down.")

app = FastAPI(
    title="Smart Comply Screening API",
    version="1.0.0",
    description="AML/CFT screening: watchlist matching + adverse media classification.",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} completed",
        extra={
            "extra": {
                "method":     request.method,
                "path":       request.url.path,
                "status":     response.status_code,
                "latency_ms": latency_ms,
            }
        }
    )
    return response

class ScreenRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=200,
        examples=["Aleksandr Petrov"],
        description="Full name of the entity to screen.",
    )

class ScreenResponse(BaseModel):
    query: str
    watchlist_hits: list[dict]
    adverse_media:  list[dict]

@app.post("/screen", response_model=ScreenResponse)
async def screen(body: ScreenRequest):
    logger.info("Screen request received", extra={"extra": {"query": body.query}})
    result = screen_entity(body.query)
    adverse_count = sum(1 for m in result["adverse_media"] if m["adverse"])
    logger.info(
        "Screen request completed",
        extra={
            "extra": {
                "query":          body.query,
                "watchlist_hits": len(result["watchlist_hits"]),
                "adverse_flags":  adverse_count,
            }
        }
    )
    return result

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "model_version":   "1.0.0",
        "embedding_model": "text-embedding-3-small",
    }