import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

GPT_DATA_DIR = os.environ.get("GPT_DATA_DIR", os.path.join(DATA_DIR, "gpt"))
LOCAL_STOCK_LIST = os.path.join(DATA_DIR, "stock_list.csv")
MARKET_CAP_PATH = os.path.join(GPT_DATA_DIR, "market_cap.csv")
