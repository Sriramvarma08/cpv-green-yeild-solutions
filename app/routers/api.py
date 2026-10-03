# app/routers/api.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import random
import numpy as np

from ..database import get_db
from ..models import Farm, Crop, Prediction
from ..schemas import (
    FarmCreate, FarmResponse, CropCreate, CropResponse,
    PredictionRequest, PredictionResponse
)

router = APIRouter()

# ─── Crop type / soil type encoding maps ───
CROP_MAP = {"rice": 0, "wheat": 1, "corn": 2, "sugarcane": 3, "cotton": 4, "soybean": 5}
SOIL_MAP = {"alluvial": 0, "black cotton": 1, "red laterite": 2, "sandy loam": 3, "clay": 4}


# ─── Farms ───
@router.get("/farms", response_model=list[FarmResponse])
def list_farms(db: Session = Depends(get_db)):
    return db.query(Farm).all()

@router.post("/farms", response_model=FarmResponse)
def create_farm(farm: FarmCreate, db: Session = Depends(get_db)):
    db_farm = Farm(**farm.model_dump())
    db.add(db_farm)
    db.commit()
    db.refresh(db_farm)
    return db_farm

@router.get("/farms/{farm_id}", response_model=FarmResponse)
def get_farm(farm_id: int, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")
    return farm


# ─── Crops ───
@router.get("/crops", response_model=list[CropResponse])
def list_crops(farm_id: int = None, db: Session = Depends(get_db)):
    query = db.query(Crop)
    if farm_id:
        query = query.filter(Crop.farm_id == farm_id)
    return query.all()


# ─── Predictions ───
@router.get("/predictions", response_model=list[PredictionResponse])
def list_predictions(db: Session = Depends(get_db)):
    return db.query(Prediction).order_by(Prediction.generated_at.desc()).limit(20).all()


@router.post("/predict")
async def predict_yield(request: Request, data: PredictionRequest, db: Session = Depends(get_db)):
    """Run yield prediction using the ML model"""
    predictor = request.app.state.yield_predictor
    
    # Encode categorical features
    crop_encoded = CROP_MAP.get(data.crop_type.lower(), 0)
    soil_encoded = SOIL_MAP.get(data.soil_type.lower(), 0)
    
    features = [
        data.temperature,
        data.humidity,
        data.rainfall,
        data.ph_level,
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.area_hectares,
        soil_encoded,
        crop_encoded,
    ]
    
    # Get prediction
    predicted_yield = predictor.predict_yield(features)
    confidence = predictor.get_confidence(features)
    recommendations = predictor.get_recommendations(features, predicted_yield)
    risk_factors = predictor.get_risk_factors(features)
    
    # Save prediction to database (use first farm or create a virtual one)
    farm = db.query(Farm).first()
    if farm:
        prediction = Prediction(
            farm_id=farm.id,
            crop_type=data.crop_type,
            predicted_yield=predicted_yield,
            confidence=confidence,
            recommendations=recommendations,
            risk_factors=risk_factors,
        )
        db.add(prediction)
        db.commit()
    
    return {
        "predicted_yield": predicted_yield,
        "confidence": confidence,
        "recommendations": recommendations,
        "risk_factors": risk_factors,
        "total_estimated_production": round(predicted_yield * data.area_hectares, 2),
        "crop_type": data.crop_type,
        "area_hectares": data.area_hectares,
    }


# ─── Dashboard Stats ───
@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    farms = db.query(Farm).all()
    crops = db.query(Crop).all()
    predictions = db.query(Prediction).order_by(Prediction.generated_at.desc()).limit(30).all()
    
    # Yield trend data (last 12 months)
    months = []
    yields_data = []
    for i in range(11, -1, -1):
        dt = datetime.now() - timedelta(days=i*30)
        months.append(dt.strftime("%b"))
        # Simulate seasonal yield variation
        base = 4.5 + 1.5 * np.sin(2 * np.pi * (dt.month - 3) / 12)
        yields_data.append(round(base + random.uniform(-0.5, 0.5), 2))
    
    # Crop distribution
    crop_counts = {}
    for crop in crops:
        crop_counts[crop.type] = crop_counts.get(crop.type, 0) + 1
    
    # Soil health data
    soil_metrics = {
        "pH Level": round(random.uniform(6.0, 7.0), 1),
        "Nitrogen": round(random.uniform(40, 80), 0),
        "Phosphorus": round(random.uniform(20, 50), 0),
        "Potassium": round(random.uniform(100, 200), 0),
        "Organic Matter": round(random.uniform(2.0, 4.5), 1),
        "Moisture": round(random.uniform(25, 60), 0),
    }
    
    # Weather data
    weather = {
        "temperature": round(random.uniform(22, 32), 1),
        "humidity": round(random.uniform(40, 75), 0),
        "rainfall": round(random.uniform(0, 15), 1),
        "wind_speed": round(random.uniform(5, 20), 1),
        "condition": random.choice(["Sunny", "Partly Cloudy", "Cloudy", "Light Rain"]),
        "forecast": [
            {"day": "Today", "high": 32, "low": 24, "condition": "Sunny"},
            {"day": "Tomorrow", "high": 30, "low": 23, "condition": "Partly Cloudy"},
            {"day": "Day 3", "high": 28, "low": 22, "condition": "Light Rain"},
            {"day": "Day 4", "high": 31, "low": 24, "condition": "Sunny"},
            {"day": "Day 5", "high": 33, "low": 25, "condition": "Sunny"},
        ]
    }
    
    return {
        "total_farms": len(farms),
        "total_area": round(sum(f.size_hectares for f in farms), 1),
        "total_crops": len(crops),
        "active_crops": sum(1 for c in crops if c.status == "growing"),
        "avg_yield": round(sum(p.predicted_yield for p in predictions) / max(len(predictions), 1), 2),
        "avg_confidence": round(sum(p.confidence for p in predictions) / max(len(predictions), 1) * 100, 1),
        "yield_trend": {"months": months, "yields": yields_data},
        "crop_distribution": crop_counts,
        "soil_health": soil_metrics,
        "weather": weather,
        "recent_predictions": [
            {
                "crop_type": p.crop_type,
                "predicted_yield": p.predicted_yield,
                "confidence": round(p.confidence * 100, 1),
                "date": p.generated_at.strftime("%Y-%m-%d") if p.generated_at else "N/A"
            }
            for p in predictions[:5]
        ],
    }
