from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from web_api_polymarket import router as polymarket_router
from fastapi.responses import JSONResponse
from log_loader import load_execution_history

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"

app = FastAPI(title="Neural Network for Crypto API", version="1.0.0")

# BUG FIX: Actually register the Polymarket router so /polymarket/* endpoints work
app.include_router(polymarket_router)


def read_csv(name: str) -> pd.DataFrame:
    path = LOGS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@app.get("/")
def root():
    return {
        "name": "Neural Network for Crypto API",
        "mode": "paper-trading / public-data only",
        "endpoints": [
            "/health",
            "/markets",
            "/signals",
            "/trades",
            "/whales",
            "/alerts",
            "/analytics",
            "/backtest",
            "/dataset",
            "/polymarket/capabilities",
            "/polymarket/status",
            "/polymarket/health",
        ],
    }


@app.get("/health")
def health():
    files = {
        "markets": (LOGS_DIR / "markets.csv").exists(),
        "signals": (LOGS_DIR / "signals.csv").exists(),
        "trades": (LOGS_DIR / "execution_log.csv").exists() or (LOGS_DIR / "daily_summary.txt").exists(),
        "whales": (LOGS_DIR / "whales.csv").exists(),
        "alerts": (LOGS_DIR / "alerts.csv").exists(),
        "system_health": (LOGS_DIR / "system_health.csv").exists(),
        "analytics": (LOGS_DIR / "trader_analytics.csv").exists(),
        "backtest": (LOGS_DIR / "backtest_summary.csv").exists(),
        "dataset": (LOGS_DIR / "historical_dataset.csv").exists(),
    }
    return {"status": "ok", "logs_dir": str(LOGS_DIR), "files": files}


@app.get("/markets")
def markets(limit: int = 50):
    df = read_csv("markets.csv")
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.tail(limit).to_dict(orient="records"))


@app.get("/signals")
def signals(limit: int = 50):
    df = read_csv("signals.csv")
    if df.empty:
        return JSONResponse([])
    if "confidence" in df.columns:
        df = df.sort_values(by="confidence", ascending=False)
    return JSONResponse(df.head(limit).to_dict(orient="records"))


@app.get("/trades")
def trades(limit: int = 50):
    df = load_execution_history(str(LOGS_DIR))
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.tail(limit).to_dict(orient="records"))


@app.get("/whales")
def whales(limit: int = 25):
    df = read_csv("whales.csv")
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.head(limit).to_dict(orient="records"))


@app.get("/alerts")
def alerts(limit: int = 50):
    df = read_csv("alerts.csv")
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.tail(limit).to_dict(orient="records"))


@app.get("/analytics")
def analytics(limit: int = 25):
    df = read_csv("trader_analytics.csv")
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.head(limit).to_dict(orient="records"))


@app.get("/backtest")
def backtest():
    df = read_csv("backtest_summary.csv")
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/dataset")
def dataset(limit: int = 100):
    df = read_csv("historical_dataset.csv")
    if df.empty:
        return JSONResponse([])
    return JSONResponse(df.tail(limit).to_dict(orient="records"))
