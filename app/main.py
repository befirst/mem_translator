import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import memes
from app.core.config import get_settings
from app.services.telegram_bot import TelegramBot

settings = get_settings()

# Initialize Telegram bot
bot = TelegramBot()

app = FastAPI(
    title=settings.PROJECT_NAME,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memes.router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """Start the Telegram bot when the FastAPI application starts"""
    await bot.start_polling()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the Telegram bot when the FastAPI application shuts down"""
    await bot.stop()

@app.get("/")
async def root():
    return {"message": "Welcome to Meme Translator API"}
