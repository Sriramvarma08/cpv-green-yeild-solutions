import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self, data_path):
        """Train the model using historical data"""
        # Load data
        data = pd.read_csv(data_path)
        X = data.drop('yield', axis=1)
        y = data['yield']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model MSE: {mse}")

        # Save model
        joblib.dump(self.model, "app/ml_models/yield_model.pkl")
        return mse
