import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.meme import Meme
from app.schemas.meme import MemeResponse
from app.services.meme_processor import MemeProcessor

router = APIRouter()
meme_processor = MemeProcessor()


@router.post("/", response_model=MemeResponse)
async def create_meme(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Process and translate a new meme
    """
    try:
        contents = await file.read()

        # Process the image
        original_text = meme_processor.extract_text(contents)
        translated_text = meme_processor.translate_text(original_text)
        processed_image = meme_processor.overlay_text(
            contents,
            translated_text,
        )

        # Upload images to S3
        filename = f"memes/{uuid.uuid4()}"
        original_url = meme_processor.upload_to_s3(
            contents,
            f"{filename}_original.jpg",
        )
        translated_url = meme_processor.upload_to_s3(
            processed_image, f"{filename}_translated.jpg"
        )

        # Create database entry
        db_meme = Meme(
            original_image_url=original_url,
            translated_image_url=translated_url,
            original_text=original_text,
            translated_text=translated_text,
        )
        db.add(db_meme)
        db.commit()
        db.refresh(db_meme)

        return db_meme
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[MemeResponse])
def get_memes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve all processed memes
    """
    memes = db.query(Meme).offset(skip).limit(limit).all()
    return memes


@router.get("/{meme_id}", response_model=MemeResponse)
def get_meme(meme_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific meme by ID
    """
    meme = db.query(Meme).filter(Meme.id == meme_id).first()
    if meme is None:
        raise HTTPException(status_code=404, detail="Meme not found")
    return meme
