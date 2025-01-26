from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MemeBase(BaseModel):
    original_text: Optional[str] = None
    translated_text: Optional[str] = None


class MemeCreate(MemeBase):
    pass


class MemeResponse(MemeBase):
    id: int
    original_image_url: str
    translated_image_url: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
