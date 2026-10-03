# app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import datetime

from .database import Base

class Farm(Base):
    __tablename__ = "farms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    size_hectares = Column(Float, nullable=False)
    soil_type = Column(String, nullable=False)
    latitude = Column(Float, default=0.0)
    longitude = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    crops = relationship("Crop", back_populates="farm", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="farm", cascade="all, delete-orphan")

class Crop(Base):
    __tablename__ = "crops"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"))
    type = Column(String, nullable=False)  # wheat, corn, rice, etc.
    variety = Column(String)
    planting_date = Column(DateTime(timezone=True))
    harvest_date = Column(DateTime(timezone=True))
    expected_yield = Column(Float)  # tons per hectare
    actual_yield = Column(Float)  # tons per hectare
    status = Column(String, default="growing")  # planned, growing, harvested
    
    # Relationships
    farm = relationship("Farm", back_populates="crops")

class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"))
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    temperature = Column(Float)  # Celsius
    humidity = Column(Float)  # Percentage
    precipitation = Column(Float)  # Millimeters
    wind_speed = Column(Float)  # km/h
    solar_radiation = Column(Float)  # MJ/m²
    condition = Column(String, default="Clear")

class SoilData(Base):
    __tablename__ = "soil_data"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"))
    measured_at = Column(DateTime(timezone=True), server_default=func.now())
    ph_level = Column(Float)
    nitrogen = Column(Float)  # kg/ha
    phosphorus = Column(Float)  # kg/ha
    potassium = Column(Float)  # kg/ha
    organic_matter = Column(Float)  # Percentage
    moisture = Column(Float)  # Percentage

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"))
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    crop_type = Column(String, nullable=False)
    predicted_yield = Column(Float)  # tons per hectare
    confidence = Column(Float)  # 0-1 value
    recommendations = Column(JSON)  # JSON array of recommendations
    risk_factors = Column(JSON)  # JSON array of risk factors
    
    # Relationships
    farm = relationship("Farm", back_populates="predictions")