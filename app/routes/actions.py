from utils.set_env import set_env

set_env()

import json
import logging
from fastapi import APIRouter, Request, Depends
from ..dependencies import get_api_key
from ..models import *
from utils.notion_helper import build_observe_data, create_observe

router = APIRouter()

@router.post("/notion/observe/create")
async def notion_observe_create(request: Request):
    try:
        body = await request.body()
        logging.info(f"Received raw body: {body.decode('utf-8')}")
        
        data = await request.json()
        logging.info(f"Received JSON data: {data}")

        records = build_observe_data(data['content'], data['time'], data['pusher'])
        for record in records:
            create_observe(record)

        return {"success": True }
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON")
        return {"success": False, "error": "Invalid JSON"}

@router.get("/notion/observe/update")
async def notion_observe_update():
    return { "success": False }