import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from fastapi import FastAPI
from route import api_router
import uvicorn

app = FastAPI()

app.include_router(api_router)
#이거 나중에 nginx사용하면 바꿔야댐
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
