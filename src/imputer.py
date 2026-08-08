
import pandas as pd 
import numpy as np 
import logging
import os 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
# Avval buni yozish kerak sababi ItertativeImputerni aktivlashtirish uchun

# Keyin esa buni yozamiz
from sklearn.impute import (
    SimpleImputer,
    KNNImputer
)
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


TARGET = "user_score"
class DataPreProcessor:
    def __init__(self, full_df: pd.DataFrame):
        # Originalini saqlab qolish 
        self.full_df = full_df.copy()
        

    # Feature engineering bosqichi 
    def feature_engineering(self):
        # release_date ustida ishlash 
        try:

            self.full_df["release_date"] = pd.to_datetime(self.full_df["release_date"])

            self.full_df["release_year"] = self.full_df["release_date"].dt.year
            self.full_df["release_month"] = self.full_df["release_date"].dt.month
            self.full_df["movie_age"] = 2026 - self.full_df["release_year"]

            self.full_df["genre_count"] = (
                self.full_df["genres"]
                .fillna("")
                .apply(lambda x: len([g for g in x.split(",") if g.strip()]))
            )

            # Muhim deb topilmagan ustunlanri tashlab yuborish 
            self.full_df.drop(
                columns=[
                "movie_id",
                "url", 
                "title",
                "release_date",
                "tmdb_movie"
                ],
                inplace=True
            )

            # logging.info(f"FE BOSQICHI MUVAFFAQIYATLI AMALGA OSHDI. COLLAR: {self.full_df.columns.tolist()}")
            # logging.info(f"SHU BILAN BIRGA UNING SHAPE: {self.full_df.shape}")

            # # engineering qilingan datasetni engineered_data folderiga saqlash
            # root_path = "/Users/murodjongafforov/Desktop/mp_last_project/data/feature_engineered"
            # self.full_df.to_csv(
            #     f"{root_path}/fe_dataset.csv", 
            #     index=False
            # )
            # logging.info(f"FE DATASET: {root_path}, MUVAFFAQQIYATLI SAQLANDI.")

            return self.full_df

        except Exception as e:
            logging.error(f"AFSUSKI FE BOSQICHI MUVAFFAQIYATLI AMAGA OSHMADI: {e}")
            raise 
        
    

    # Train va Test split qilish bosqichi 
    def split_data(self):

        X = self.full_df.drop(self.TARGET, axis=1)
        y = self.full_df[self.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test





# Handling Missing Valuelar uchun Advanced Missing Value (Imputation)larni tadbiq qilish.

class Imputer:
    def __init__(self, method):
        self.method = method 
        self.numerical_imputer = self._get_numerical_imputer()
        self.categorical_imputer = self._get_categorical_imputer()

    def _get_numerical_imputer(self):
        try:
            if self.method == "mean":
                imputer = SimpleImputer(
                    strategy = "mean"
                )
            elif self.method == "median":
               imputer = SimpleImputer(
                    strategy = "median"
                )
            elif self.method == "mice":
                imputer = IterativeImputer(
                    max_iter = 10,
                    random_state = 42
                )
            elif self.method == "knn":
                imputer = KNNImputer(
                    n_neighbors = 5
                )
            else:
                raise ValueError(
                    f"Nomalum imputation yaratildi: {self.method}"
                )
            logging.info(f"{self.method} imputer saralandi")
            return imputer
        except Exception as e:
            logging.exception(f"Imputer yaratish muvaffaqqiyatsizlikka uchradi: {e}")
            raise 

    # Kategorik ustundagi qiymatlar uchun imputation
    def _get_categorical_imputer(self):
        try:
            cat_imputer = SimpleImputer(
                strategy = "most_frequent"
            )
            
            logging.info(f"Categorical ustunlar imputation yaratildi")
            return cat_imputer
        except Exception as e:
            logging.exception(f"Categorical qiymatlar bilan yaratish muvaffaqiyatsizlikka uchradi: {e}")
            raise 
        

    # X_train uchun fit_transform funksiyasini orqali missing qiymatlarni to'ldiramiz
    def fit_transform(self, X_train):
        try:
            # Tartiblangan original ustunlar
            self.feature_order = X_train.columns.tolist()

            # Raqamli ustunlarni olish
            self.numeric_cols = X_train.select_dtypes(include = [np.number]).columns.tolist()
            logging.info(f"{len(self.numeric_cols)} raqamli ustunlar saralandi.")
            # Objectli ustunlarni olish
            self.cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            logging.info(f"{len(self.cat_cols)} obyektli ustunlar saralandi.")

            X_train_num = self.numerical_imputer.fit_transform(
                X_train[self.numeric_cols]
            )
            X_train_cat = self.categorical_imputer.fit_transform(
                X_train[self.cat_cols]
            )
            # DataFramega o'tkazib olamiz
            X_train_num = pd.DataFrame(
                X_train_num,
                columns = self.numeric_cols,
                index = X_train.index
            )
            X_train_cat = pd.DataFrame(
                X_train_cat,
                columns = self.cat_cols,
                index = X_train.index
            )
            # Birlashtiramiz X_trainga
            X_train = pd.concat(
                [X_train_num, X_train_cat],
                axis = 1
            )  
            X_train = X_train[self.feature_order]
            logging.info("Train data muvaffaqiyatli transform qilindi.")    
            return X_train
        
        except Exception as e:
            logging.exception(f"Train data fit_transform() vaqtida muvaffaqiyatsizlikka uchradi: {e}")
            raise  # bu dasturni to'xtadadi agar muammo aiqlansa

    # X_test uchun transform qilish orqali missing qiymatlarni to'ldiramiz
    def transform(self, X_test):
        try:
            X_test_num = self.numerical_imputer.transform(
                X_test[self.numeric_cols]
            )
            X_test_cat = self.categorical_imputer.transform(
                X_test[self.cat_cols]
            )
            
            # DataFramega o'tkazamiz
            X_test_num = pd.DataFrame(
                X_test_num,
                columns = self.numeric_cols,
                index = X_test.index
            )
            X_test_cat = pd.DataFrame(
                X_test_cat,
                columns = self.cat_cols,
                index = X_test.index
            )
            # Birlashtiramiz X_test_num va X_test_cat ustunlarnini
            X_test = pd.concat(
                [X_test_num, X_test_cat],
                axis = 1
            )
            X_test = X_test[self.feature_order]   
            logging.info("Test data muvaffaqiyatli transform qilindi.")
            return X_test
        except Exception as e:
            logging.exception(f"Test data transform() vaqtida muvaffaqiyatsizlikka uchradi: {e}")
            raise 
        

            







        

