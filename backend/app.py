from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from predict import predict_bird


app = FastAPI(
    title="BirdSense-AI API",
    description="AI-powered bird species identification from audio",
    version="1.0.0"
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "BirdSense-AI API is running!",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {extension}"
        )

    # Create unique temporary filename
    filename = f"{uuid.uuid4()}{extension}"
    file_path = UPLOAD_DIR / filename

    try:

        # Save uploaded audio
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run model prediction
        result = predict_bird(str(file_path))

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        # Delete temporary file
        if file_path.exists():
            file_path.unlink()