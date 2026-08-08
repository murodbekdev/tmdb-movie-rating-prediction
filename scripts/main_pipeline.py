import os 
import sys 
import logging 
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Loyihamizning asosiy folderni pathga qo'shish
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.data_loading import DataLoader
from src.transformers import fix_skewness
from src.imputer import Imputer 
from src.encoder import Encoder 
from src.scaler import Scaler 
from src.models import Algorithms 



# Project Root Path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)



# Logging sozlanmalari 
log_dir = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(
    log_dir,
    "comparison_pipeline.log"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            log_path,
            encoding="utf-8"
        )
    ],
    force=True
)

logger = logging.getLogger(__name__)

def main():
    try:
        logging.info("🚀 Asosiy ML Pipeline ishga tushdi...")
        # 1. Datasetlarni yuklash va birlashtirish 
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/feature_engineered/fe_dataset.csv"))
        logging.info(f"📂 Tayyor dataset yuklanmoqda: {data_path}")

        if not os.path.exists(data_path):
            logging.error(f"❌ Fayl topilmadi: {data_path}")
            return 

        full_df = pd.read_csv(data_path)
        logging.info(f"✅ Dataset yuklandi. Hajmi: {full_df.shape}")

        # 2. Target va featurelarni ajratish 
        target_col = "user_score"
        if target_col not in full_df.columns:
            logging.error(f"❌ Target ustun topilmadi: {target_col}")

        X = full_df.drop(columns = [target_col])
        y = full_df[target_col]

        logger.info(
            f"🎯 Target: {target_col}"
        )

        logger.info(
            f"📊 X shape: {X.shape}"
        )

        logger.info(
            f"🎯 y shape: {y.shape}"
        )

        # 3. Train va Test Split 
        logger.info(
            "✂️ Ma'lumotlar Train va Testga bo'linmoqda (80/20)..."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )

        logger.info(
            f"X_train: {X_train.shape}"
        )

        logger.info(
            f"X_test: {X_test.shape}"
        )

        # 4. Skewness qiyshiqlikni tekkislash 
        logger.info(
            "📊 Skewness tekshirilmoqda..."
        )

        X_train_skew = fix_skewness(
            X_train.copy(),
            threshold=0.75
        )

        X_test_skew = fix_skewness(
            X_test.copy(),
            threshold=0.75
        )

        logger.info(
            "✅ Skewness preprocessing tugadi."
        )

        # 5. Imputationlar usullari ro'yxati 
        imputation_methods = ["mean", "median", "mice", "knn"]

        all_results = []
        best_overall_score = -float("inf")
        best_model_obj = None
        best_model_info = {}

        # Har bir imputation uchun tsikl
        for method in imputation_methods:
            logger.info(
                "\n=========================================="
            )

            logger.info(
                f"🩹 Imputation usuli boshlandi: {method.upper()}"
            )

            logger.info(
                "=========================================="
            )

            try: 
                # Imputerlarni ishga tushurish (nusxa olib ishaltamiz)
                imputer = Imputer(method = method)
                X_train_imp = imputer.fit_transform(X_train_skew.copy())
                X_test_imp = imputer.transform(X_test_skew.copy())
                logger.info(
                    f"✅ {method} imputation tugadi."
                )

                # Encoding 
                train_encoder = Encoder(X_train_imp)
                X_train_enc, label_cols, onehot_cols = train_encoder.encodla_train(onehot_threshold = 5)
                train_columns = X_train_enc.columns

                test_encoder = Encoder(X_test_imp)
                X_test_enc = test_encoder.encodla_test(
                    label_cols = label_cols,
                    onehot_cols = onehot_cols,
                    train_columns = train_columns
                )

                logger.info(
                    f"✅ Encoding tugadi. "
                    f"Train shape: {X_train_enc.shape}, "
                    f"Test shape: {X_test_enc.shape}"
                )

                # Scaling 
                scale_obj = Scaler(X_train_enc)
                X_train_scaled, scaler, num_cols = scale_obj.scale_train()
                test_scaler_obj = Scaler(X_test_enc)
                X_test_scaled = test_scaler_obj.scale_test(scaler, num_cols)
                logger.info(
                    "✅ Scaling tugadi."
                )

                # Modellar va tuning 
                models_config = Algorithms.get_models_and_grids()
                logger.info(
                    f"🤖 {len(models_config)} ta model topildi."
                )

                for name, config in models_config.items():
                    model = config["model"]
                    params = config["params"]
                    needs_scaling = config["needs_scaling"]

                    if needs_scaling:
                        X_tr = X_train_scaled
                        X_te = X_test_scaled
                        scaling_status = "Yes"
                    else:
                        X_tr = X_train_enc
                        X_te = X_test_enc
                        scaling_status = "No"
                    logger.info(
                        f"⏳ Tuning: {name} | "
                        f"Imputer: {method} | "
                        f"Scaling: {scaling_status}"
                    )

                    if params:
                        grid_search = GridSearchCV(
                            estimator = model,
                            param_grid = params,
                            cv = 3,
                            scoring = "r2",
                            n_jobs = -1
                        )
                        grid_search.fit(X_tr, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = str(grid_search.best_params_)
                    else:
                        model.fit(X_tr, y_train)
                        best_model = model
                        best_params = "Default (No params)"

                    preds = best_model.predict(X_te)
                    r2 = r2_score(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))

                    # Natijalarni ro'yxatga qo'shish 
                    all_results.append({
                        "Imputation": method,
                        "Model": name,
                        "Scaling": scaling_status,
                        "Best_Params": best_params,
                        "R2_SCORE": round(r2, 4),
                        "RMSE": round(rmse, 4)
                    })

                    logger.info(
                        f"    → {name} ({method}) | "
                        f"R2: {r2:.4f} | "
                        f"RMSE: {rmse:.4f}"
                    )
                    # Eng yaxshi modelni topib borib boramiz 
                    if r2 > best_overall_score:
                        best_overall_score = r2
                        best_model_obj = best_model
                        best_model_info = {
                            "Imputation": method,
                            "Model": name,
                            "R2_SCORE": r2,
                            "RMSE": rmse
                        }
            except Exception as inner_e:
                logger.error(
                    f"❌ {method} imputation usulida "
                    f"xatolik yuz berdi: {str(inner_e)}",
                    exc_info=True
                )
                continue
        # Resultni DataFramega o'tkazamiz
        results_df = pd.DataFrame(
            all_results
        )


        if results_df.empty:

            logger.error(
                "❌ Hech qanday model natijasi olinmadi."
            )

            return

        # 6. Barcha natijalarni CSV qilib saqlash 
        results_df = pd.DataFrame(all_results)
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tables"))
        os.makedirs(output_dir, exist_ok = True)

        output_path = os.path.join(output_dir, "all_imputation_model_results.csv")
        results_df.to_csv(output_path, index = False, float_format = "%.4f")
        logger.info(
            f"💾 Barcha natijalar saqlandi: {output_path}"
        )

        # 7. End yaxshi modelni .pkl formatda saqlash 
        if best_model_obj is not None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
            os.makedirs(models_dir, exist_ok = True)

            model_save_path = os.path.join(models_dir, "best_improved_model.pkl")
            joblib.dump(best_model_obj, model_save_path)

            logger.info(f"🏆 ENG YAXSHI MODEL SAQLANDI: {model_save_path}")
            logger.info(f"    -Imputation: {best_model_info["Imputation"]}")
            logger.info(f"    - Model: {best_model_info["Model"]}")
            logger.info(f"    - R2 Score: {best_model_info["R2_SCORE"]:.4f}")

        # Terminalga ham natijani chiqarish 
        print("\n" + "="*100)
        print("🎯 BARCHA IMPUTATION VA MODELLARNING TAQQOSLASH NATIJALARI:")
        print("="*100)
        print(results_df.sort_values(by="R2_SCORE", ascending= False).to_string(index=False))
        print("="*100)
        print(f"🏆 ENG YAXSHI MODEL: {best_model_info['Model']} (Imputation: {best_model_info['Imputation']}, R2: {best_model_info['R2_SCORE']:.4f})")
        print(f"💾 Saqlangan manzil: models/best_improved_model.pkl")
        print("="*100)

    except Exception as e:
        logger.info(f"❌ Pipeline ichida global xatolik yuz berdi: {str(e)}", exc_info = True)


# Run qilish uchun
if __name__ == "__main__":

    main()







             






