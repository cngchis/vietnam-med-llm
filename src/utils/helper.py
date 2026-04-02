import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Missing env variable: {key}")
    return val