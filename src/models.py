import logging
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, objective


# # fillarni import qilib olish 
# # path berish 
# ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.os.append(ROOT_PATH)

# from src.

class Algorithms:
    @staticmethod
    def get_models_and_grids():
        try: 
            logging.info("⚙️ Barcha modellar va ularning giperparametrlarGrid (tarmoqlari) tayyorlanmoqda...")
            models_config = {
                "LinearRegresion": {
                    "model": LinearRegression(),
                    "params": {},
                    "needs_scaling": True
                },
                "Ridge": {
                    "model": Ridge(),
                    "params":{"params": [0.1, 1.0, 10.0]},
                    "needs_scaling": True
                },
                "Lasso":{
                    "model":Lasso(),
                    "params":{"params": [0.01, 0.1, 1.0]},
                    "needs_scaling": True
                },
                "ElasticNet": {
                    "model": ElasticNet(random_state=42),
                    "params":{"alpha":[0.1, 1.0], "l1_ratio": [0.2, 0.5]},
                    "needs_scaling": True

                },
                "SVR":{
                    "model":SVR(),
                    "params":{"C": [1, 10], "kernel":["rbf", "linear"]},
                    "needs_scaling": True
                },
                "RandomForest":{
                    "model":RandomForestRegressor(random_state=42),
                    "params": {"n_estimators": [50,100,150], "max_depth":[None, 10, 20]},
                    "needs_scaling": False
                },
                "GradientBoosting":{
                    "model":GradientBoostingRegressor(random_state=42),
                    "params":{"n_estimators": [50, 100, 150], "learning_rate":[0.05, 0.1]},
                    "needs_scaling": False
                },
                "XGBoost":{
                    "model":XGBRegressor(random_state=42),
                    "params":{"n_estimators": [50,100,150], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
                    "needs_scaling": False
                }

            }
            logging.info("⚙️ Barcha modellar va ularning giperparametrlarGrid (tarmoqlari) tayyorlanmoqda...")
            return models_config 
        
        except Exception as e:
            logging.error(f"❌ Modellar luga'atini tuzishda xatolik yuz berd: {str(e)}", exc_info= True)
            raise e