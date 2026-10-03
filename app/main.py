# app/main.py
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
import random
import os
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from .database import get_db, engine, SessionLocal, Base
from .models import Farm, Crop, Prediction, WeatherData, SoilData
from .ml_models.predictor import YieldPredictor

# Initialize services with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database and ML models
    Base.metadata.create_all(bind=engine)
    app.state.yield_predictor = YieldPredictor()
    await app.state.yield_predictor.load_model()
    
    # Seed demo data if database is empty
    db = SessionLocal()
    try:
        if db.query(Farm).count() == 0:
            _seed_demo_data(db)
    finally:
        db.close()
    
    yield
    # Shutdown: Clean up resources
    await app.state.yield_predictor.unload_model()

def _seed_demo_data(db):
    """Seed the database with demo data"""
    import random
    from datetime import datetime, timedelta
    
    farms = [
        Farm(name="Green Valley Farm", location="Punjab, India", size_hectares=45.2, 
             soil_type="Alluvial", latitude=30.73, longitude=76.78),
        Farm(name="Sunrise Acres", location="Maharashtra, India", size_hectares=32.8, 
             soil_type="Black Cotton", latitude=19.07, longitude=72.87),
        Farm(name="Golden Harvest Fields", location="Tamil Nadu, India", size_hectares=28.5, 
             soil_type="Red Laterite", latitude=11.12, longitude=78.65),
    ]
    db.add_all(farms)
    db.commit()
    
    crop_types = [
        ("Rice", "Basmati 370", "growing"),
        ("Wheat", "HD 2967", "growing"),
        ("Corn", "DHM 117", "planned"),
        ("Sugarcane", "Co 0238", "growing"),
        ("Cotton", "Bt Cotton", "harvested"),
        ("Soybean", "JS 335", "growing"),
    ]
    
    for farm in farms:
        for i, (ctype, variety, status) in enumerate(random.sample(crop_types, 3)):
            crop = Crop(
                farm_id=farm.id, type=ctype, variety=variety,
                planting_date=datetime.now() - timedelta(days=random.randint(30, 120)),
                expected_yield=round(random.uniform(2.0, 8.0), 1),
                actual_yield=round(random.uniform(1.5, 7.5), 1) if status == "harvested" else None,
                status=status
            )
            db.add(crop)
    db.commit()
    
    for farm in farms:
        for i in range(5):
            pred = Prediction(
                farm_id=farm.id,
                crop_type=random.choice(["Rice", "Wheat", "Corn"]),
                predicted_yield=round(random.uniform(2.5, 7.5), 2),
                confidence=round(random.uniform(0.7, 0.95), 2),
                generated_at=datetime.now() - timedelta(days=i*7),
                recommendations=[{"message": "Optimal conditions detected"}],
                risk_factors=[{"factor": "Low", "description": "Normal conditions"}]
            )
            db.add(pred)
    db.commit()


# Create FastAPI application with lifespan management
app = FastAPI(
    title="CPV Green Yeild Solutions",
    description="AI-powered crop yield prediction and farming recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Include API routers
from .routers import api
app.include_router(api.router, prefix="/api")


# Root endpoint - Landing page
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page"""
    return templates.TemplateResponse(request=request, name="landing.html")


# Dashboard endpoint
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db=Depends(get_db)):
    """Main dashboard page"""
    farms = db.query(Farm).all()
    total_area = sum(f.size_hectares for f in farms)
    total_crops = db.query(Crop).count()
    active_crops = db.query(Crop).filter(Crop.status == "growing").count()
    predictions = db.query(Prediction).order_by(Prediction.generated_at.desc()).limit(10).all()
    avg_yield = sum(p.predicted_yield for p in predictions) / len(predictions) if predictions else 0
    avg_confidence = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0
    
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "farms": farms,
        "total_area": round(total_area, 1),
        "total_crops": total_crops,
        "active_crops": active_crops,
        "total_farms": len(farms),
        "avg_yield": round(avg_yield, 2),
        "avg_confidence": round(avg_confidence * 100, 1),
        "predictions": predictions,
    })


# Predict page
@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Yield prediction form page"""
    return templates.TemplateResponse(request=request, name="predict.html")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "CPV Green Yeild Solutions API"
    }