import os   
import sys
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            params={
                "Decision Tree":{
                    'criterion':['gini','entropy'],
                    'max_depth':[3,5,10,None]
                },
                "Random Forest":{
                    'n_estimators':[50,100,200],
                    'criterion':['gini','entropy'],
                    'max_depth':[3,5,10,None]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05],
                    'n_estimators':[50,100,200]
                },
                "K-Neighbors Classifier":{
                    'n_neighbors':[3,5,7,9],
                    'weights':['uniform','distance'],
                    'metric':['minkowski','euclidean','manhattan']
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,.05],
                    'n_estimators':[50,100,200]
                }
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if(best_model_score < 0.6):
                raise CustomException("No best model found", sys
                                      )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

          
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
      
        except Exception as e:
            logging.error("Error occurred in model trainer")
            raise CustomException(e, sys)
        
    

