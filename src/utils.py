# If i want to read my database from dataset like MongoDB Client
# If i want to save my model over cloud, So i can write my code here for this purposes.
import os
import sys
import numpy as np
import pandas as pd
import dill # for creating a pickle file
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Saving the model
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)


# Model training
# def evaluate_models(x_train,y_train,x_test,y_test,models):
#     try:
#         report = {}
#         for i in range(len(list(models))):
#             model = list(models.values())[i]
            
#             # Train model
#             model.fit(x_train,y_train)
            
#             # Prediction
#             y_train_pred = model.predict(x_train)
#             y_test_pred = model.predict(x_test)
            
#             # R2 score
#             train_model_score = r2_score(y_train,y_train_pred)
#             test_model_score = r2_score(y_test,y_test_pred)
            
#             report[list(models.keys())[i]] = test_model_score
#         return report
#     except Exception as e:
#         raise CustomException(e,sys)



# Model Training with Hyper Parameter Tuning
def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            # GridSearchCV
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)
            
            # Set best parameters
            model.set_params(**gs.best_params_)
            
            # Train model
            model.fit(x_train,y_train)
            
            # Prediction
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            # R2 score
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)