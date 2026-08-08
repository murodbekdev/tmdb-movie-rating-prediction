import pandas as pd 
import numpy as np 
import logging 
import os, sys

# Log file pathini beramiz 
log_file = r"/Users/murodjongafforov/Desktop/mp_last_project/logs/data_loading.log"

# Log folder mavjudligini teskhiramiz
os.makedirs(os.path.dirname(log_file), exist_ok = True)

# Set-up logging, basicConfigni bir marta ishlatish kifoya.
logging.basicConfig(
    filename = log_file,
    filemode = "w",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
)

# Log uchun test harablari 
logging.info("✅ Logging ishlayapti...")
logging.error("❌ Bu xatlik testi...")
logging.warning("🚨 Bu ogohlantirish testi...")
print(f"Logging qilish muvaffaqiyatli yakunlandi. Bu yerdan tekshiring: {log_file}")

# Class uchun path 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_loading import DataLoader

loader = DataLoader(r"/Users/murodjongafforov/Desktop/mp_last_project/data/raw_data")

# hamma datasetlarni birlashtirish 
try:
    full_df = loader.datasets_concate()
    # Birlashtirilgan datasetlarni web_scrapped_data folderiga saqlash
    output_path = os.path.join(
        "/Users/murodjongafforov/Desktop/mp_last_project/data/web_scrapped_data",
        "tmdb_movies.csv"
    )
    full_df.to_csv(output_path, index=False)
    logging.info(f"✅ Muvaffaqiyatli birlashtirildi. Shape: {full_df.shape}")
except Exception as e:
    logging.warning(f"🚨 Nimadir sodir bo'ldi. Ogoh bo'ling va qaya teshiring: {e}")
