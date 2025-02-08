from fastapi import FastAPI
from route import api_router
import uvicorn

app = FastAPI()

# API 라우트 등록
app.include_router(api_router)
#이거 나중에 nginx사용하면 바꿔야댐
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
