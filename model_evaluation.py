import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from data_preprocessing import DataPreprocessor

class ModelEvaluator:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.load_data()
        self.load_models()
    
    def load_data(self):
        print(" Loading and Preprocessing Data...")
        (self.X_train_class, self.X_test_class, self.y_train_class, self.y_test_class, 
         self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg) = self.preprocessor.run_preprocessing()
    
    def load_models(self):
        print(" Loading Trained Models...")
        self.classification_model = pickle.load(open("models/classification.pkl", "rb"))
        self.regression_model = pickle.load(open("models/regression.pkl", "rb"))
        self.clustering_model = pickle.load(open("models/clustering.pkl", "rb"))
    
    def evaluate_classification(self):
        print("\n Evaluating Classification Model...")
        y_pred_class = self.classification_model.predict(self.X_test_class)
        
        metrics = {
            "Accuracy": accuracy_score(self.y_test_class, y_pred_class),
            "Precision": precision_score(self.y_test_class, y_pred_class),
            "Recall": recall_score(self.y_test_class, y_pred_class),
            "F1-Score": f1_score(self.y_test_class, y_pred_class),
            "ROC-AUC": roc_auc_score(self.y_test_class, y_pred_class)
        }
        
        for metric, value in metrics.items():
            print(f"🔹 {metric}: {value:.2f}")
    
    def evaluate_regression(self):
        print("\n Evaluating Regression Model...")
        y_pred_reg = self.regression_model.predict(self.X_test_reg)
        
        metrics = {
            "MAE": mean_absolute_error(self.y_test_reg, y_pred_reg),
            "MSE": mean_squared_error(self.y_test_reg, y_pred_reg),
            "RMSE": np.sqrt(mean_squared_error(self.y_test_reg, y_pred_reg)),
            "R² Score": r2_score(self.y_test_reg, y_pred_reg)
        }
        
        for metric, value in metrics.items():
            print(f"🔹 {metric}: {value:.2f}")
    
    def evaluate_clustering(self):
        print("\n Evaluating Clustering Model...")
        X_cluster = self.X_train_class.copy()
        cluster_labels = self.clustering_model.predict(X_cluster)
        
        metrics = {
            "Silhouette Score": silhouette_score(X_cluster, cluster_labels),
            "Davies-Bouldin Index": davies_bouldin_score(X_cluster, cluster_labels)
        }
        
        for metric, value in metrics.items():
            print(f"🔹 {metric}: {value:.2f}")
    
    def run_evaluation(self):
        self.evaluate_classification()
        self.evaluate_regression()
        self.evaluate_clustering()
        print("\n Model Evaluation Completed!")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()