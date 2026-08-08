# DATASETLARNI BIRLASHIRISH

import pandas as pd 
import numpy as np
import logging 
import os, sys 


class DataLoader():
    def __init__(self, path):
        self.path = path 

    def datasets_concate(self):
        # datasets_id 
        tmdb_movies = ["1", "2", "3", "4", "5", "6", "7", "8"]
        df_list = []

        for tmdb_movie in tmdb_movies:
            filename = f"tmdb_movies_{tmdb_movie}.csv"
            file_path = os.path.join(self.path, filename)
            # Agar file mavjud bo'lsa, uni saqlab olamiz.
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['tmdb_movie'] = tmdb_movie
                    df_list.append(df)
                    logging.info(f"Dataset yuklandi: {file_path}")
                except Exception as e:
                    logging.error(f"Yuklash jarayonida xatolik: {file_path} - {e}")
            else:
                logging.warning(f"File topilmadi: {file_path}, tashlab o'tib ket.")

        # Hamma datasetlarni concatenate orqali qo'shib olamiz
        if df_list:
            try:
                full_df = pd.concat(df_list, ignore_index = True)
                logging.info(f"To'liq dataset yuklandi. Ko'rinishi: {full_df.shape}")
                return full_df
            except Exception as e:
                logging.error(f"Dataframeni qo'shishda xatlik bo'ldi: {e}")
                return pd.DataFrame()
        else:
            logging.error("Hech qanday data yuklanmadi! Iltimos, file yo'lini va folder yo'lini tekshiring!")
            return pd.DataFrame()



