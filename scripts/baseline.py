import pandas as pd
import numpy as np 
import os, sys
import logging 
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error


# Modellar
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor

# Pathni berishim kerak 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoder import Encoder
from src.scaler import Scaler


# --LOGGING SOZLANMALARI--
baseline_log_path = r"/Users/murodjongafforov/Desktop/mp_last_project/logs/baseline_model.log"
logging.basicConfig(
    # filename = baseline_log_path,
    level = logging.INFO,
    format = "%(asctime)s - [%(levelname)s] - %(message)s",
    handlers = [
        logging.StreamHandler(), # Terminalga chiqarish
        logging.FileHandler(baseline_log_path, encoding = "utf-8")
    ]
)
try: 
    logging.info("🚀 Baseline skripti ishga tushdi...")

    # 1. Ma'lumotni yuklash 
    data_path = 'data/web_scrapped_data/tmdb_movies.csv'
    df = pd.read_csv(data_path)
    logging.info(f"📂 Ma'lumot yuklanmoqda: {data_path}")


    # --- TEKSHIRUV: Ustunlar nomini ko'rib olamiz ---
    print("📊 Mavjud ustunlar:", df.columns.tolist())
    print("🎯 user_score ustunidagi dastlabki 5 ta qiymat:", df["user_score"].head(5).values)
    df["user_score"] = pd.to_numeric(df["user_score"], errors='coerce')
    df = df.dropna(subset=["user_score"])

   # Target va Feature'larni ajratish
    y = df["user_score"].dropna()  # Targetda NaN bo'lsa, o'sha qatorni olib tashlaymiz
    
    # X ni ham y ga moslab qisqartiramiz (faqat y da bor qatorlarni qoldiramiz)
    X = df.loc[y.index].drop(
        columns=[
            "movie_id",
            "url",
            "title",
            "release_date",
            "tmdb_movie",
            "user_score",
            "overview"
        ],
        errors = "ignore"    
    )
    # Sonli va kategorik ustunlarni avtomatik ajratib olish:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    logging.info("✂️ Ma'lumot Train (80%) va Test (20%) qismlarga bo'linmoqda...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )
    # Numerical va Categorical ustunlarni aniqlash
    logging.info("🧹 Bo'sh joylar (Missing values) to'ldirilmoqda...")


    X_train_full = X_train.copy()
    X_test_full = X_test.copy()
    # Sonli ustunlar uchun ("median")
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy = "median")
        X_train_full[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test_full[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
    # Matnli ustunlar uchun eng ko'p uchraydigan so'z ("most_frequent")
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy = "most_frequent")
        X_train_full[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test_full[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        
    # Encoding, matnlarni raqamga o'tkazish: X_train_enc va X_test_enc
    logging.info("🔢 Categorical ustunlar encoded qilinmoqda...")
    train_encoder = Encoder(X_train_full)
    X_train_enc, label_cols, onehot_cols = train_encoder.encodla_train(onehot_threshold = 5)

    test_encoder = Encoder(X_test_full)
    X_test_enc = test_encoder.encodla_test(label_cols,onehot_cols, X_train_enc.columns)

    # Scaling o'lchamlarini tenglashtirish 
    logging.info("⚖️ Scaling jarayoni bajarilmoqda...")
    train_scaler = Scaler(X_train_enc)
    # Train uchun scaling 
    X_train_scaled, fitted_scaler, num_cols = train_scaler.scale_train()
    # Test uchun scaling 
    test_scaler = Scaler(X_test_enc)
    X_test_scaled = test_scaler.scale_test(fitted_scaler, num_cols)

    X_train_scaled = X_train_scaled.select_dtypes(include=[np.number])
    X_test_scaled = X_test_scaled.select_dtypes(include=[np.number])
    X_train_enc = X_train_enc.select_dtypes(include=[np.number])
    X_test_enc = X_test_enc.select_dtypes(include=[np.number])


    # Modellar: Bazi Algoritmalrda scaling qilish kerak bazilarida esa shart emas..
    models_config = {
        "LinearRegression": {
            "model":LinearRegression(),
            "needs_scaling": True
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=42),
            "needs_scaling": False
        },
        "RandomForest":{
            "model": RandomForestRegressor(random_state=42),
            "needs_scaling": False
        },
        "XGBRegressor": {
            "model": XGBRegressor(random_state=42),
            "needs_scaling": False
        }
    }

    # natijalarni saqlash uchun bo'sh ro'yxat
    results_list = []

    # Modellarni sinovdan o'tkazish 
    logging.info("🤖 Modellar o'qitilib, bashorat qilinmoqda...")
    for name, config in models_config.items():
        model = config["model"]
        needs_scaling = config["needs_scaling"]

        # needs_scalingga qarab ma'lumotni tanlaymiz
        if needs_scaling:
            X_tr = X_train_scaled
            X_te = X_test_scaled
            scaling_text = "Yes"
        else:
            X_tr = X_train_enc
            X_te = X_test_enc
            scaling_text = "No"

        # Modelni o'qitish va bashorat qilish 
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)

        # Metriklar
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Natijalarni ro'yxatga qo'shamiz 
        results_list.append({
            "Model" : name,
            "Scaling": scaling_text,
            "R2_SCORE": r2,
            "RMSE": rmse
        })
        logging.info(f"   -> {name} tugatildi (R2: {r2:.4f}, RMSE: {rmse:.4f})")

    # Natijalarni DataFrame qilish orqali CSV filega saqlash
    results_df = pd.DataFrame(results_list)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(base_dir, "results", "tables")
    #results/tables folderiga uni saqlashdan oldin tekshiramiz
    os.makedirs(output_dir, exist_ok = True)
    # output_path orqali CSV filemizni saqlaymiz
    output_path = os.path.join(output_dir, "baseline_model_results.csv")
    results_df.to_csv(output_path, index=False, float_format="%.4f") # index = False orqali default index tiymatlarni bermsadan table chiroyli chiqishini taminlaymiz.
    logging.info(f"💾 Natijalar muvaffaqiyatli saqlandi: {output_path}")

    print("\n" + "="*65)
    print("🎯 BASELINE YAKUNIY NATIJALARI:")
    print("="*65)
    print(results_df.to_string(index=False))
    print("="*65)

except Exception as e:
    logging.error(f"❌ Xatolik yuz berdi: {str(e)}", exc_info = True)




