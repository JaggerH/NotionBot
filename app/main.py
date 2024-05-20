from fastapi import FastAPI
from .routes import actions
from utilities.set_env import set_env

set_env()
app = FastAPI(title="Notion Bot", servers=[{"url": "https://notion-bot.azurewebsites.net"}])

app.include_router(actions.router)

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})