import pandas as pd 
import numpy as np 
import logging 

def fix_skewness(df:pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Raqamli ustunlarni skewnessini tekshirish va log1p orqali tekkislash"""
    try: 
        df_copy = df.copy()
        numerical_cols = df_copy.select_dtypes(include = np.number).columns

        skewed_cols = df_copy[numerical_cols].apply(lambda x: x.skew()).abs()
        skewed_cols = skewed_cols[skewed_cols > threshold].index

        for col in skewed_cols:
            if df_copy[col].min() >= 0:
                df_copy[col] = np.log1p(df_copy[col])
        logging.info(f"{len(df_copy.columns)} - ustunlar ustida skeweness muvaffiqiyatli amalga oshirildi ✅.")
        return df_copy
    except Exception as e:
        logging.error(f"🚨 Afsuski skeweness bosqichida muammo yuzaga keldi: {e}")
        raise
