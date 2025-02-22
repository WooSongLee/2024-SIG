from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from route import api_router
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)
app.include_router(api_router)
#이거 나중에 nginx사용하면 바꿔야댐
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8888)
