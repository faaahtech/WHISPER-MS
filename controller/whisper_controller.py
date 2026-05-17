from config.settings import settings
import whisper

import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool


class InvalidAudioFormatError(Exception):
    pass


class WhisperTranscriptionError(Exception):
    pass


class WhisperController:
    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".mp4", ".flac"}

    ALLOWED_CONTENT_TYPES = {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/m4a",
        "audio/flac",
        "video/mp4",
        "application/octet-stream",
    }

    def __init__(self):
        model_name = settings.WHISPER_MODEL
        self.model = whisper.load_model(model_name)

    def _get_file_suffix(self, audio_file: UploadFile) -> str:
        return Path(audio_file.filename or "").suffix.lower() or ".tmp"

    def _validate_file(self, audio_file: UploadFile) -> None:
        suffix = self._get_file_suffix(audio_file)

        is_valid_content_type = audio_file.content_type in self.ALLOWED_CONTENT_TYPES
        is_valid_extension = suffix in self.ALLOWED_EXTENSIONS

        if not is_valid_content_type and not is_valid_extension:
            raise InvalidAudioFormatError(
                "Formato inválido. Envie um áudio mp3, wav, m4a, webm, ogg, mp4 ou flac."
            )

    def _save_temp_file(self, audio_file: UploadFile, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(audio_file.file, temp_file)
            return temp_file.name

    async def transcribe_audio(
        self,
        audio_file: UploadFile,
        language: str = "pt",
        task: Literal["transcribe", "translate"] = "transcribe",
    ) -> dict:
        temp_path = None

        try:
            self._validate_file(audio_file)

            suffix = self._get_file_suffix(audio_file)

            await audio_file.seek(0)

            temp_path = await run_in_threadpool(
                self._save_temp_file,
                audio_file,
                suffix,
            )

            result = await run_in_threadpool(
                self.model.transcribe,
                temp_path,
                language=language,
                task=task,
                fp16=False,
            )

            return {
                "filename": audio_file.filename,
                "language": result.get("language"),
                "task": task,
                "text": result.get("text", "").strip(),
                "segments": [
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                    }
                    for segment in result.get("segments", [])
                ],
            }

        except InvalidAudioFormatError:
            raise

        except Exception as error:
            raise WhisperTranscriptionError(str(error))

        finally:
            await audio_file.close()

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)