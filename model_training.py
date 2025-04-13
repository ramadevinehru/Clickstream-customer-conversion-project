import pickle
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from data_preprocessing import DataPreprocessor

class MLPipeline:
    def __init__(self):
        self.classification_models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        }
        self.regression_models = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.clustering_models = {
            "K-Means": MiniBatchKMeans(n_clusters=3, batch_size=1000, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            "Hierarchical": AgglomerativeClustering(n_clusters=3, linkage='average')
        }
    
    def train_models(self, X_train_class, X_test_class, y_train_class, y_test_class, X_train_reg, y_train_reg):
        """Train models and save the best ones."""
        print("Training Classification Models...")
        best_class_model = None
        best_acc = 0
        for name, model in self.classification_models.items():
            model.fit(X_train_class, y_train_class)
            acc = model.score(X_test_class, y_test_class)
            print(f"{name} Accuracy: {acc:.2f}")
            if acc > best_acc:
                best_acc = acc
                best_class_model = model
        pickle.dump(best_class_model, open("models/classification.pkl", "wb"))

        print("\nTraining Regression Models...")
        best_reg_model = None
        best_r2 = -1
        for name, model in self.regression_models.items():
            model.fit(X_train_reg, y_train_reg)
            r2 = model.score(X_train_reg, y_train_reg)
            print(f"{name} R² Score: {r2:.2f}")
            if r2 > best_r2:
                best_r2 = r2
                best_reg_model = model
        pickle.dump(best_reg_model, open("models/regression.pkl", "wb"))

        print("\nTraining Clustering Models...")
        X_cluster_sample = X_train_class.sample(n=5000, random_state=42)
        kmeans_model = self.clustering_models["K-Means"]
        kmeans_model.fit(X_cluster_sample)
        pickle.dump(kmeans_model, open("models/clustering.pkl", "wb"))

# Run the training pipeline
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train_class, X_test_class, y_train_class, y_test_class, X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocessor.run_preprocessing()

    pipeline = MLPipeline()
    pipeline.train_models(X_train_class, X_test_class, y_train_class, y_test_class, X_train_reg, y_train_reg)
