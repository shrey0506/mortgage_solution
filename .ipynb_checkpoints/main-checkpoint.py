from fastapi import FastAPI
from mortgage.app.view.view import router

app = FastAPI()

app.include_router(router)