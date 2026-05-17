from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from controller.whisper_controller import (
    InvalidAudioFormatError,
    WhisperController,
    WhisperTranscriptionError,
)


router = APIRouter(prefix="/whisper", tags=["Whisper"])

whisper_controller = WhisperController()


@router.post("/send/audio", status_code=200)
async def new_order(
    audio_file: UploadFile = File(...),
    language: str = Form("pt"),
    task: Literal["transcribe", "translate"] = Form("transcribe"),
):
    try:
        transcription = await whisper_controller.transcribe_audio(
            audio_file=audio_file,
            language=language,
            task=task,
        )

        return {
            "success": True,
            "message": "Áudio transcrito com sucesso.",
            "data": transcription,
        }

    except InvalidAudioFormatError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except WhisperTranscriptionError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao transcrever o áudio: {str(error)}",
        )