# Here we are going to implement the code for the model training......
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting the train and test input array")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            # evaluate_model in utils
            # we will get the score in the dictionary
            model_report:dict=evaluate_models(x_train = x_train, y_train = y_train, x_test=x_test, y_test=y_test, models=models)
            
            # To get best model from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            
            logging.info("Best model found on both training and testing dataset.")
            
            save_object(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(x_test)
            R2_score = r2_score(y_test,predicted)
            
            return R2_score
            
        except Exception as e:
            raise CustomException(e,sys)