from fastapi import FastAPI

app = FastAPI(title="langgraph-ai-agent")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
