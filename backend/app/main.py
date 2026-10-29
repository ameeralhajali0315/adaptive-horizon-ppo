from fastapi import FastAPI, HTTPException

from backend.app.results_service import get_research_results

app = FastAPI(
    title="Adaptive-Horizon PPO Research API",
    description="Backend API for the PPO research dashboard",
    version="0.2.0",
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "application": "Adaptive-Horizon PPO Research API",
    }


@app.get("/api/results")
def research_results():
    try:
        return get_research_results()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
