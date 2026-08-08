import pandas as pd 
import numpy as np
import logging 
from sklearn.preprocessing import MinMaxScaler



class Scaler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    # scale train uchun 
    def scale_train(self):
        try: 
            num_cols = self.df.select_dtypes(include = np.number).columns.tolist()

            # Scalerni initalize qilish va train qismiga fit qilish bosqichi
            scaler = MinMaxScaler()
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
            logging.info("Train uchun muvaffaqiyatli Scaling amalga oshirildi.")
            return self.df, scaler, num_cols
        except Exception as e:
            logging.error("Afsus, Test uchun Scaling muvaffaqiyatsizlikka uchradi.")
            raise e
    # scale test uchun 
    def scale_test(self, scaler, num_cols):
        try: 
            self.df[num_cols] = scaler.transform(self.df[num_cols])
            logging.info("Train uchun muvaffaqiyatli Scaling amalga oshirildi.")
            return self.df
        except Exception as e:
            logging.error("Afsus, Train uchun Scaling muvaffaqiyatsizlikka uchradi.")
            raise e



# # Ishlatish: Train fit_transform()
# X_train, scaler, num_cols = scale_train(X_train)

# # Ishlatish: Test transform()
# X_test = scale_test(X_test, scaler, num_cols)


