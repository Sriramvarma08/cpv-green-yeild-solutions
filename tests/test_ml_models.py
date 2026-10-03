# tests/test_ml_models.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, patch
from app.ml_models.predictor import YieldPredictor
from app.ml_models.trainer import ModelTrainer

@pytest.mark.asyncio
async def test_yield_predictor_initialization():
    """Test that the yield predictor initializes correctly"""
    predictor = YieldPredictor()
    assert predictor.model is None
    assert predictor.preprocessor is None
    assert predictor.feature_columns is not None

@pytest.mark.asyncio
async def test_model_training():
    """Test model training with sample data"""
    # Create sample training data
    sample_data = pd.DataFrame({
        'crop_type': ['wheat', 'corn', 'rice'] * 10,
        'soil_type': ['clay', 'silt', 'sand'] * 10,
        'temperature': np.random.uniform(15, 30, 30),
        'precipitation': np.random.uniform(0, 100, 30),
        'humidity': np.random.uniform(30, 80, 30),
        'ph_level': np.random.uniform(5.5, 7.5, 30),
        'nitrogen': np.random.uniform(5, 50, 30),
        'phosphorus': np.random.uniform(5, 40, 30),
        'potassium': np.random.uniform(50, 300, 30),
        'organic_matter': np.random.uniform(1, 5, 30),
        'solar_radiation': np.random.uniform(10, 30, 30),
        'yield': np.random.uniform(1, 5, 30)
    })
    
    predictor = YieldPredictor()
    score = await predictor.train_model(sample_data)
    
    assert score is not None
    assert predictor.model is not None
    assert predictor.preprocessor is not None

@pytest.mark.asyncio
async def test_yield_prediction():
    """Test yield prediction with sample input"""
    # Mock a trained model
    predictor = YieldPredictor()
    with patch.object(predictor, 'model', AsyncMock()), \
         patch.object(predictor, 'preprocessor', AsyncMock()):
        
        # Mock the preprocessor transform method
        predictor.preprocessor.transform.return_value = np.random.rand(1, 10)
        
        # Mock the model predict method
        predictor.model.predict.return_value = [3.5]
        
        sample_input = {
            'crop_type': 'wheat',
            'soil_type': 'clay',
            'temperature': 22.5,
            'precipitation': 45.0,
            'humidity': 65.0,
            'ph_level': 6.5,
            'nitrogen': 25.0,
            'phosphorus': 15.0,
            'potassium': 150.0,
            'organic_matter': 2.5,
            'solar_radiation': 20.5
        }
        
        result = await predictor.predict_yield(sample_input)
        
        assert 'predicted_yield' in result
        assert 'confidence' in result
        assert 'recommendations' in result
        assert result['predicted_yield'] == 3.5

@pytest.mark.asyncio
async def test_recommendation_generation():
    """Test recommendation generation logic"""
    predictor = YieldPredictor()
    
    sample_input = {
        'crop_type': 'wheat',
        'ph_level': 5.8,  # Low pH
        'nitrogen': 10.0,  # Low nitrogen
        'precipitation': 30.0,
        'temperature': 25.0
    }
    
    recommendations = predictor._generate_recommendations(sample_input, 2.0)
    
    # Should have recommendations for pH and nitrogen
    assert len(recommendations) >= 2
    assert any('ph' in rec['category'] for rec in recommendations)
    assert any('nitrogen' in rec['message'] for rec in recommendations)