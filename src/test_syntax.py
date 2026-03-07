try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import time
    import json

    app = FastAPI(title=\"Test\")
    
    @app.get(\"/\")
    async def root():
        return {\"test\": \"ok\"}
    
    print(\"Syntax OK\")
except Exception as e:
    print(f\"Error: {e}\")
