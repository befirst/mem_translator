# Meme Translator

A service that automatically translates memes from English to Russian using OCR, translation APIs, and image processing.

## Features

- Telegram bot integration for receiving and sending memes
- OCR text extraction from images using Tesseract with enhanced accuracy
- Improved image processing techniques for better text visibility
- **Image processing with text overlay**: Not yet implemented.
- **Storage of memes and translations in PostgreSQL**: Not yet implemented.
- **Cloud storage integration for images**: Not yet implemented.
- **Admin panel for monitoring and management**: Not yet implemented.
- Text translation using Google Translate API

## Development Setup

### Prerequisites

1. Install pyenv (for Python version management):
   ```bash
   # On macOS using Homebrew
   brew install pyenv
   ```

2. Install Python using pyenv:
   ```bash
   pyenv install 3.13.0
   pyenv local 3.13.0
   ```

3. Install Poetry (for dependency management):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

### Project Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mem_translator
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

5. Run database migrations:
   ```bash
   alembic upgrade head
   ```

### Running the Application

Start the application using:
```bash
poetry run uvicorn app.main:app --reload
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
```bash
# For macOS
brew install tesseract

# For Ubuntu
sudo apt-get install tesseract-ocr
```

3. Set up environment variables in `.env`:
```
TELEGRAM_BOT_TOKEN=your_bot_token
DATABASE_URL=postgresql://user:password@localhost/dbname
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your_bucket_name
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
mem_translator/
├── app/
│   ├── api/            # API routes
│   ├── core/           # Core functionality
│   ├── db/             # Database models and operations
│   ├── services/       # Business logic
│   └── main.py         # FastAPI application
├── tests/              # Test files
├── alembic/            # Database migrations
├── pyproject.toml      # Project configuration
├── README.md           # Project documentation
└── Dockerfile          # Docker configuration
```

## License

MIT
