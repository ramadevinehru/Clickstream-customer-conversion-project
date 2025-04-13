
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, file_path="E:/clickstream project/train_data.csv"):
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load the dataset."""
        self.df = pd.read_csv(self.file_path)
    
    def encode_categorical_features(self):
        """Perform mapping and encoding for categorical data."""
        mappings = {
            "country": {1: "Australia", 2: "Austria", 3: "Belgium", 4: "British Virgin Islands", 5: "Cayman Islands", 
                        6: "Christmas Island", 7: "Croatia", 8: "Cyprus", 9: "Czech Republic", 10: "Denmark", 11: "Estonia", 
                        12: "unidentified", 13: "Faroe Islands", 14: "Finland", 15: "France", 16: "Germany", 17: "Greece", 
                        18: "Hungary", 19: "Iceland", 20: "India", 21: "Ireland", 22: "Italy", 23: "Latvia", 24: "Lithuania", 
                        25: "Luxembourg", 26: "Mexico", 27: "Netherlands", 28: "Norway", 29: "Poland", 30: "Portugal", 
                        31: "Romania", 32: "Russia", 33: "San Marino", 34: "Slovakia", 35: "Slovenia", 36: "Spain", 
                        37: "Sweden", 38: "Switzerland", 39: "Ukraine", 40: "United Arab Emirates", 41: "United Kingdom", 
                        42: "USA", 43: "biz", 44: "com", 45: "int", 46: "net", 47: "org"},
            "colour": {1: "beige", 2: "black", 3: "blue", 4: "brown", 5: "burgundy", 6: "gray", 7: "green", 
                       8: "navy blue", 9: "of many colors", 10: "olive", 11: "pink", 12: "red", 13: "violet", 14: "white"},
            "location": {1: "top left", 2: "top middle", 3: "top right", 4: "bottom left", 5: "bottom middle", 6: "bottom right"},
            "page1_main_category": {1: "trousers", 2: "skirts", 3: "blouses", 4: "sale"}
        }
        for col, mapping in mappings.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)

        # Convert Boolean Columns to Integers (0/1)
        bool_cols = self.df.select_dtypes(include=['bool']).columns
        self.df[bool_cols] = self.df[bool_cols].astype(int)

        # Label Encoding for binary columns
        binary_cols = ['model_photography', 'price_2']
        le = LabelEncoder()
        for col in binary_cols:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col])

        # One-hot encoding for categorical variables
        categorical_cols = ['country', 'page1_main_category', 'colour', 'location', 'page2_clothing_model']
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

        # Convert all boolean columns to integers
        bool_cols_after_encoding = self.df.select_dtypes(include=['bool']).columns
        self.df[bool_cols_after_encoding] = self.df[bool_cols_after_encoding].astype(int)
    
    def feature_engineering(self):
        """Generate new features."""
        self.df["session_length"] = self.df.groupby("session_id")["order"].transform("count")

    def scale_features(self):
        """Apply feature scaling."""
        numerical_cols = ["session_length", "price"]
        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
    
    def balance_dataset(self):
        """Apply SMOTE to balance the classification target variable."""
        X = self.df.drop(columns=['price_2'])
        y = self.df['price_2']
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def split_data(self):
        """Split data into training and testing sets for classification and regression."""
        X_class, y_class = self.balance_dataset()
        X_reg = self.df.drop(columns=['price_2'])  # Keep price column for regression
        y_reg = self.df['price']

        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        return X_train_class, X_test_class, y_train_class, y_test_class, X_train_reg, X_test_reg, y_train_reg, y_test_reg

    def run_preprocessing(self):
        self.load_data()
        self.encode_categorical_features()
        self.feature_engineering()
        self.scale_features()
        
        # Save the processed dataset
        self.df.to_csv("processed_data.csv", index=False)

        return self.split_data()