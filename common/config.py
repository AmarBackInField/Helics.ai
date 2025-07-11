import os


DEFAULT_MAX_RESULTS=2

# Google Search Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

MODEL="gpt-4.1-mini"