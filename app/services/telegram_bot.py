from loguru import logger
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.core.config import get_settings
from app.services.meme_processor import MemeProcessor
from io import BytesIO

settings = get_settings()


class TelegramBot:
    def __init__(self):
        self.meme_processor = MemeProcessor()
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.application = None
        self._running = False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        await update.message.reply_text(
            "Hi! Send me a meme and I will translate it to Russian."
        )

    async def process_image(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """Process received images."""
        try:
            # Get the photo with highest resolution
            photo = update.message.photo[-1]
            logger.debug(f"Received photo with file_id: {photo.file_id}")
            
            # Download the photo
            photo_file = await context.bot.get_file(photo.file_id)
            logger.debug(f"Retrieved file info: {photo_file}")
            
            # Download as bytes
            photo_bytes = await photo_file.download_as_bytearray()
            logger.debug(f"Downloaded photo, size: {len(photo_bytes)} bytes")

            # Extract text from image
            original_text, top_results = self.meme_processor.extract_text(photo_bytes)
            if not original_text:
                await update.message.reply_text(
                    "Sorry, I couldn't find any text in this image."
                )
                return

            logger.info(f"Best extracted text: {original_text}")

            # Format top 3 results message
            confidence_msg = "Top 3 detected texts:\n" + "\n".join([
                f"{i+1}. {text} (confidence: {conf:.1f}%)"
                for i, (text, method, conf) in enumerate(top_results[:3])
            ])

            # Translate best text
            translated_text = self.meme_processor.translate_text(original_text)
            logger.info(f"Translated text: {translated_text}")

            # Process image
            processed_image = self.meme_processor.overlay_text(
                photo_bytes, translated_text
            )

            # Convert bytes to BytesIO for Telegram
            bio = BytesIO(processed_image)
            bio.name = 'translated_meme.jpg'

            # Send the processed image and results
            await update.message.reply_text(confidence_msg)
            await update.message.reply_photo(
                photo=bio,
                caption=(
                    f"Best match: {original_text}\n"
                    f"Translated text: {translated_text}"
                ),
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            await update.message.reply_text(
                "Sorry, there was an error processing your image."
            )

    async def start_polling(self):
        """Start the bot polling"""
        if self._running:
            return

        try:
            # Create application
            builder = ApplicationBuilder()
            builder.token(self.token)
            self.application = builder.build()

            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(
                MessageHandler(filters.PHOTO, self.process_image)
            )

            # Start the bot
            logger.info("Starting Telegram bot...")
            self._running = True

            # Initialize but don't start polling yet
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()

        except Exception as e:
            self._running = False
            logger.error(f"Error running Telegram bot: {e}")
            raise

    async def stop(self):
        """Stop the bot"""
        if not self._running:
            return

        try:
            if self.application:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                self._running = False
                logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
            raise
