"""Entry point for the Multi-Agent Startup Evaluation System."""
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
