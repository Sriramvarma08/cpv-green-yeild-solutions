# tests/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.database import get_db, Base, engine
from app.models import Farm, Crop, Prediction

# Create test client
client = TestClient(app)

# Override database dependency for testing
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_database():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop all tables after tests
    Base.metadata.drop_all(bind=engine)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_dashboard_page():
    """Test dashboard page loads correctly"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_farm_summary_endpoint():
    """Test farm summary API endpoint"""
    # First create a test farm
    farm_data = {
        "name": "Test Farm",
        "location": "Test Location",
        "size_hectares": 10.5,
        "soil_type": "loam"
    }
    response = client.post("/api/v1/farms/", json=farm_data)
    assert response.status_code == 200
    
    # Now test the summary endpoint
    response = client.get("/api/v1/farms/summary")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Test Farm" in response.text

@patch("app.routers.dashboard.WeatherService")
def test_weather_endpoint(MockWeatherService):
    """Test weather API endpoint with mocked service"""
    mock_service = MockWeatherService.return_value
    mock_service.get_forecast.return_value = {
        "location": "Test Location",
        "current": {
            "temp_c": 22.5,
            "humidity": 65,
            "precipitation": 0.2,
            "wind_speed": 12.3,
            "condition": "Partly Cloudy"
        }
    }
    
    response = client.get("/api/v1/weather/current")
    assert response.status_code == 200
    assert "22.5" in response.text
    assert "Partly Cloudy" in response.text

def test_prediction_endpoint_no_data():
    """Test predictions endpoint with no data"""
    response = client.get("/api/v1/predictions/latest")
    assert response.status_code == 200
    assert "No predictions" in response.text

@patch("app.ml_models.predictor.YieldPredictor.predict_yield")
def test_prediction_creation(MockPredictYield):
    """Test prediction creation endpoint"""
    MockPredictYield.return_value = {
        "predicted_yield": 3.5,
        "confidence": 0.85,
        "recommendations": [],
        "timestamp": "2023-01-01T00:00:00"
    }
    
    # First create a test farm
    farm_data = {
        "name": "Test Farm",
        "location": "Test Location",
        "size_hectares": 10.5,
        "soil_type": "loam"
    }
    farm_response = client.post("/api/v1/farms/", json=farm_data)
    farm_id = farm_response.json()["id"]
    
    # Create prediction
    prediction_data = {
        "farm_id": farm_id,
        "crop_type": "wheat",
        "predicted_yield": 3.5,
        "confidence": 0.85,
        "recommendations": [],
        "risk_factors": []
    }
    
    response = client.post("/api/v1/predictions/", json=prediction_data)
    assert response.status_code == 200
    assert response.json()["predicted_yield"] == 3.5