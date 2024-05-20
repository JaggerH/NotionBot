import os
from dotenv import load_dotenv

def set_env():
    if os.getenv("ENV") != "product":
        # 在开发环境中加载 .env 文件
        load_dotenv(override=True)