# app/ml_models/predictor.py
import joblib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class YieldPredictor:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yield_model.pkl")

    async def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                self._train_default_model()
        except Exception:
            self._train_default_model()
    
    def _train_default_model(self):
        """Create and train a default model with realistic crop data"""
        np.random.seed(42)
        n_samples = 500
        
        # Features: temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium, area, soil_encoded, crop_encoded
        temperature = np.random.uniform(15, 35, n_samples)
        humidity = np.random.uniform(30, 90, n_samples)
        rainfall = np.random.uniform(50, 300, n_samples)
        ph = np.random.uniform(4.5, 8.5, n_samples)
        nitrogen = np.random.uniform(10, 140, n_samples)
        phosphorus = np.random.uniform(5, 145, n_samples)
        potassium = np.random.uniform(5, 205, n_samples)
        area = np.random.uniform(1, 100, n_samples)
        soil_type = np.random.randint(0, 5, n_samples)
        crop_type = np.random.randint(0, 6, n_samples)
        
        X = np.column_stack([temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium, area, soil_type, crop_type])
        
        # Realistic yield formula (tons per hectare)
        y = (
            2.0
            + 0.05 * nitrogen
            + 0.03 * phosphorus
            + 0.02 * potassium
            + 0.1 * rainfall / 100
            - 0.5 * np.abs(ph - 6.5)
            - 0.02 * np.abs(temperature - 25)
            + np.random.normal(0, 0.3, n_samples)
        )
        y = np.clip(y, 0.5, 12.0)
        
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    async def unload_model(self):
        """Unload the model"""
        self.model = None

    def predict_yield(self, features: list) -> float:
        """Predict crop yield based on features
        
        Features order: [temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium, area, soil_encoded, crop_encoded]
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        prediction = self.model.predict([features])[0]
        return round(max(0.1, prediction), 2)
    
    def get_confidence(self, features: list) -> float:
        """Estimate prediction confidence based on feature completeness and ranges"""
        # Simple heuristic for confidence
        temp, humidity, rainfall, ph, n, p, k, area, soil, crop = features
        
        confidence = 0.85
        # Penalize extreme values
        if temp < 10 or temp > 40:
            confidence -= 0.1
        if ph < 4.0 or ph > 9.0:
            confidence -= 0.1
        if rainfall < 20 or rainfall > 400:
            confidence -= 0.05
        
        return round(max(0.3, min(0.98, confidence)), 2)
    
    def get_recommendations(self, features: list, predicted_yield: float) -> list:
        """Generate farming recommendations based on input features"""
        temp, humidity, rainfall, ph, nitrogen, phosphorus, potassium, area, soil, crop = features
        
        recommendations = []
        
        # pH recommendations
        if ph < 5.5:
            recommendations.append({
                "category": "Soil Health",
                "priority": "high",
                "icon": "⚠️",
                "message": f"Soil pH is acidic ({ph}). Apply agricultural lime at 2-3 tons/hectare to raise pH to 6.0-6.5.",
                "impact": "+15-20% yield potential"
            })
        elif ph > 7.5:
            recommendations.append({
                "category": "Soil Health",
                "priority": "high",
                "icon": "⚠️",
                "message": f"Soil pH is alkaline ({ph}). Apply elemental sulfur or ammonium sulfate to lower pH.",
                "impact": "+10-15% yield potential"
            })
        else:
            recommendations.append({
                "category": "Soil Health",
                "priority": "low",
                "icon": "✅",
                "message": f"Soil pH ({ph}) is in the optimal range. Continue monitoring.",
                "impact": "Maintaining current yield"
            })
        
        # Nitrogen recommendations
        if nitrogen < 40:
            recommendations.append({
                "category": "Fertilization",
                "priority": "high",
                "icon": "🌿",
                "message": f"Nitrogen level ({nitrogen} kg/ha) is low. Apply urea or ammonium nitrate at {int(80 - nitrogen)} kg/ha.",
                "impact": "+20-30% yield potential"
            })
        elif nitrogen > 120:
            recommendations.append({
                "category": "Fertilization",
                "priority": "medium",
                "icon": "⚡",
                "message": f"Nitrogen level ({nitrogen} kg/ha) is excessive. Reduce application to prevent environmental runoff.",
                "impact": "Cost savings & sustainability"
            })
        
        # Water management
        if rainfall < 80:
            recommendations.append({
                "category": "Irrigation",
                "priority": "high",
                "icon": "💧",
                "message": f"Low rainfall ({rainfall}mm). Set up drip irrigation or schedule supplemental watering of 30-50mm/week.",
                "impact": "+25-35% yield potential"
            })
        elif rainfall > 250:
            recommendations.append({
                "category": "Drainage",
                "priority": "medium",
                "icon": "🌊",
                "message": f"High rainfall ({rainfall}mm). Ensure proper field drainage to prevent waterlogging and root rot.",
                "impact": "Prevents 15-25% yield loss"
            })
        
        # Temperature
        if temp > 35:
            recommendations.append({
                "category": "Climate",
                "priority": "high",
                "icon": "🌡️",
                "message": f"High temperature ({temp}°C). Consider shade nets, mulching, and increased irrigation frequency.",
                "impact": "Prevents heat stress damage"
            })
        elif temp < 15:
            recommendations.append({
                "category": "Climate",
                "priority": "medium",
                "icon": "❄️",
                "message": f"Low temperature ({temp}°C). Consider row covers or delayed planting until conditions improve.",
                "impact": "Prevents frost damage"
            })
        
        # Potassium
        if potassium < 50:
            recommendations.append({
                "category": "Fertilization",
                "priority": "medium",
                "icon": "🧪",
                "message": f"Potassium level ({potassium} kg/ha) is low. Apply muriate of potash at {int(100 - potassium)} kg/ha.",
                "impact": "+10-15% yield potential"
            })
        
        return recommendations
    
    def get_risk_factors(self, features: list) -> list:
        """Identify risk factors from the input conditions"""
        temp, humidity, rainfall, ph, nitrogen, phosphorus, potassium, area, soil, crop = features
        
        risks = []
        
        if humidity > 80:
            risks.append({
                "factor": "High Humidity",
                "level": "high",
                "description": "Increased risk of fungal diseases (blight, mildew). Apply preventive fungicide."
            })
        
        if temp > 35 and humidity > 70:
            risks.append({
                "factor": "Heat Stress",
                "level": "critical",
                "description": "Combined high temperature and humidity can cause severe crop stress and reduced pollination."
            })
        
        if nitrogen > 100 and rainfall > 200:
            risks.append({
                "factor": "Nutrient Leaching",
                "level": "medium",
                "description": "High rainfall may wash away applied nitrogen. Consider split application."
            })
        
        if ph < 5.0:
            risks.append({
                "factor": "Aluminum Toxicity",
                "level": "high",
                "description": "Very acidic soil may release toxic aluminum ions. Immediate liming recommended."
            })
        
        if rainfall < 50:
            risks.append({
                "factor": "Drought Stress",
                "level": "high",
                "description": "Insufficient rainfall may cause wilting and reduced grain fill."
            })
        
        if not risks:
            risks.append({
                "factor": "Low Risk",
                "level": "low",
                "description": "Current conditions are favorable. Continue monitoring for any changes."
            })
        
        return risks
