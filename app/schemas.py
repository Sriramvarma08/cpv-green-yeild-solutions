# app/schemas.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Farm schemas
class FarmBase(BaseModel):
    name: str
    location: str
    size_hectares: float
    soil_type: str
    latitude: float = 0.0
    longitude: float = 0.0

class FarmCreate(FarmBase):
    pass

class FarmResponse(FarmBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Crop schemas
class CropBase(BaseModel):
    farm_id: int
    type: str
    variety: Optional[str] = None
    planting_date: Optional[datetime] = None
    harvest_date: Optional[datetime] = None
    expected_yield: Optional[float] = None
    actual_yield: Optional[float] = None
    status: str = "growing"

class CropCreate(CropBase):
    pass

class CropResponse(CropBase):
    id: int

    class Config:
        from_attributes = True

# Prediction schemas
class PredictionBase(BaseModel):
    farm_id: int
    crop_type: str
    predicted_yield: float
    confidence: float
    recommendations: Optional[List[dict]] = None
    risk_factors: Optional[List[dict]] = None

class PredictionCreate(PredictionBase):
    pass

class PredictionResponse(PredictionBase):
    id: int
    generated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Prediction request (for the form)
class PredictionRequest(BaseModel):
    crop_type: str
    soil_type: str
    area_hectares: float
    temperature: float
    humidity: float
    rainfall: float
    ph_level: float
    nitrogen: float
    phosphorus: float
    potassium: float
