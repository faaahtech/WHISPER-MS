import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

settings = Settings()