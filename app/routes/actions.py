from fastapi import APIRouter, Depends
from ..dependencies import get_api_key
from ..models import *

router = APIRouter()

@router.get("/notion/observe/create")
async def notion_observe_create():
    return { "success": True }

@router.get("/notion/observe/update")
async def notion_observe_update():
    return { "success": False }