import os 
import sys 
import logging 
import pandas as pd




# Proyekt yo'li 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.imputer import DataPreProcessor

# Log file pathini berish 
log_file = r"/Users/murodjongafforov/Desktop/mp_last_project/logs/imputer.log"
os.makedirs(os.path.dirname(log_file), exist_ok = True)

# Set-up qilish loggingni 
logging.basicConfig(
    filename = log_file,
    filemode = "w",
    level = logging.INFO,
    format = "%(asctime)s -%(levelname)s - %(filename)s - %(message)s" 
)

logging.info("✅ Logging ishlayapti...")
logging.error("❌ Bu xatolik testi..")

# Datasetni o'qish 
input_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "web_scrapped_data",
    "tmdb_movies.csv"
)

df_fe = pd.read_csv(input_path)

# Feature Engineering 
processor = DataPreProcessor(df_fe)
df_fe = processor.feature_engineering()
# Saqlash 
try: 
    output_path = os.path.join(
        "data",
        "feature_engineered",
        "fe_dataset.csv"
    )

    df_fe.to_csv(output_path, index=False)
    logging.info(f"cMuvaffaqiyatli Feature Engineered dataset saqlandi. Shape: {output_path}")
    print(df_fe.head())

except Exception as e:
    logging.warning(f"🚨 Nimadir sodir bo'ldi. Ogoh bo'ling va qaya teshiring: {e}")


# # Featur Engineered datasetni saqlash
# try:
#     full_df = pd.read_csv("../data/web_scrapped_data/tmdb_movies.csv")
#     processor = DataPreProcessor(full_df)
#     df_fe = processor.feature_engineering()
#     logging.info(f"✅ Muvaffaqiyatli Feature Engineered dataset saqlandi. Shape: {df_fe.shape}")
# except Exception as e:
#     logging.warning(f"🚨 Nimadir sodir bo'ldi. Ogoh bo'ling va qaya teshiring: {e}")




































# import pandas as pd 
# import numpy as np 
# import logging 
# import os, sys


# # Log file pathini berish 
# log_file = r"/Users/murodjongafforov/Desktop/mp_last_project/logs/imputer.log"
# os.makedirs(os.path.dirname(log_file), exist_ok = True)

# # Set-up qilish loggingni 
# logging.basicConfig(
#     filename = log_file,
#     filemode = "w",
#     level = logging.INFO,
#     format = "%(asctime)s -%(levelname)s - %(filename)s - %(message)s" 
# )

# logging.info("✅ Logging ishlayapti...")
# logging.error("❌ Bu xatolik testi..")



# # Avval log fileni pathini berib olamiz
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# from src.imputer import DataPreProcessor



# # Featur Engineered datasetni saqlash
# try:
#     full_df = pd.read_csv("../data/web_scrapped_data/tmdb_movies.csv")
#     processor = DataPreProcessor(full_df)
#     df_fe = processor.feature_engineering()
#     logging.info(f"✅ Muvaffaqiyatli Feature Engineered dataset saqlandi. Shape: {df_fe.shape}")
# except Exception as e:
#     logging.warning(f"🚨 Nimadir sodir bo'ldi. Ogoh bo'ling va qaya teshiring: {e}")







