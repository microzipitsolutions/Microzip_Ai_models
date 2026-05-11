import os
from dotenv import load_dotenv

load_dotenv()

# --- Cricbuzz & Market Configuration ---
CRICBUZZ_ID  = "152108"
MARKET_ID    = "1.257942254"

# --- API Keys ---
# Backend team should set these in a .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SATRADAR_TOKEN = os.getenv("SATRADAR_TOKEN")

# --- File Paths ---
MODEL_PATH    = "./data/models/pattern_model_v2.pkl"
HISTORICAL_DB = "./data/parsed/cricsheet_all.csv"
DETAILED_LOG  = "./live/detailed_ball_log.json"
PRICE_FILE    = "./live/ws_price.json"

