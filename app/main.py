
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.endpoints import hair  # endpoints.hair 모듈 불러오기

# FastAPI 앱 초기화
app = FastAPI(title=settings.PROJECT_NAME)
app.mount("/static", StaticFiles(directory="static"), name="static")

# hair 라우터 등록
app.include_router(hair.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
