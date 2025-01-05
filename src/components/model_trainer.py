import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                
                "Decision Tree": DecisionTreeClassifier(),
                "KNeighbour": KNeighborsClassifier(),
                "Guassian": GaussianNB(),
                "BernoulliNB": BernoulliNB()
    
            }
            param_grids = {
               'DecisionTree' : {
                'criterion': ['gini', 'entropy'],
                'max_depth': [2, 3, 4, 5, 6, 7, 8],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
          },

           'KNeighbour' : {
                    'n_neighbors': [ 7, 9, 11, 13, 15,25, 34],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },

           'Guass' : {
                    'var_smoothing': np.logspace(0,-9, num=100)
                },

                'Bern' : {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]
                }
            }
        
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=param_grids)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            



            
        except Exception as e:
            raise CustomException(e,sys)