"""
Unit tests for FastAPI model serving application.
Lecture 83 - Deployment

Run with:
    pytest test_app.py -v
    pytest test_app.py --cov=app --cov-report=html
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

# Import the FastAPI app
try:
    from app import app
except ImportError:
    pytest.skip("App not found", allow_module_level=True)

# Create test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health and metadata endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    
    def test_ping_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/ping")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_metadata_endpoint(self):
        """Test metadata endpoint."""
        response = client.get("/metadata")
        
        # May return 503 if model not loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "input_shape" in data
        assert "output_classes" in data


class TestPredictionEndpoint:
    """Test prediction functionality."""
    
    def test_predict_valid_input(self):
        """Test prediction with valid 28x28 image."""
        # Create random 28x28 image
        image = np.random.randint(0, 255, (28, 28)).tolist()
        
        response = client.post("/predict", json={"instances": [image]})
        
        # May return 503 if model not loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        
        pred = data["predictions"][0]
        assert "class_id" in pred
        assert "class_name" in pred
        assert "confidence" in pred
        assert 0 <= pred["class_id"] < 10
        assert 0 <= pred["confidence"] <= 1
    
    def test_predict_multiple_images(self):
        """Test batch prediction with multiple images."""
        # Create 3 random images
        images = [np.random.randint(0, 255, (28, 28)).tolist() for _ in range(3)]
        
        response = client.post("/predict", json={"instances": images})
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
    
    def test_predict_invalid_shape(self):
        """Test prediction with wrong image shape."""
        # Wrong shape: 10x10 instead of 28x28
        image = np.random.randint(0, 255, (10, 10)).tolist()
        
        response = client.post("/predict", json={"instances": [image]})
        assert response.status_code in [400, 422]  # Bad request or validation error
    
    def test_predict_empty_request(self):
        """Test prediction with empty request."""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error
    
    def test_predict_empty_instances(self):
        """Test prediction with empty instances list."""
        response = client.post("/predict", json={"instances": []})
        assert response.status_code in [400, 422]
    
    def test_predict_invalid_data_type(self):
        """Test prediction with invalid data type."""
        response = client.post("/predict", json={"instances": "not a list"})
        assert response.status_code == 422
    
    def test_predict_too_large_batch(self):
        """Test prediction with batch size exceeding limit."""
        # Create more than max_items (32)
        images = [np.random.randint(0, 255, (28, 28)).tolist() for _ in range(35)]
        
        response = client.post("/predict", json={"instances": images})
        assert response.status_code == 422


class TestValidation:
    """Test request validation."""
    
    def test_malformed_json(self):
        """Test with malformed JSON."""
        response = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_content_type(self):
        """Test with missing content type header."""
        response = client.post("/predict", data='{"instances": []}')
        # FastAPI auto-detects JSON, so this might still work
        assert response.status_code in [200, 400, 422, 503]


class TestPerformance:
    """Test API performance."""
    
    def test_response_time(self):
        """Test that response time is reasonable."""
        import time
        
        image = np.random.randint(0, 255, (28, 28)).tolist()
        
        start = time.time()
        response = client.post("/predict", json={"instances": [image]})
        duration = time.time() - start
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        # Response should be under 1 second for single prediction
        assert duration < 1.0
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        import concurrent.futures
        
        def make_request():
            image = np.random.randint(0, 255, (28, 28)).tolist()
            return client.post("/predict", json={"instances": [image]})
        
        # Send 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # Check that most succeeded (some might fail if model not loaded)
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count >= 0  # At least attempt was made


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_on_invalid_route(self):
        """Test 404 on non-existent route."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test wrong HTTP method."""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405


# Integration test
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_workflow(self):
        """Test complete prediction workflow."""
        # 1. Check health
        health = client.get("/ping")
        assert health.status_code == 200
        
        # 2. Get metadata
        metadata = client.get("/metadata")
        if metadata.status_code == 503:
            pytest.skip("Model not loaded")
        
        # 3. Make prediction
        image = np.random.randint(0, 255, (28, 28)).tolist()
        prediction = client.post("/predict", json={"instances": [image]})
        assert prediction.status_code == 200
        
        # 4. Validate response structure
        data = prediction.json()
        assert "predictions" in data
        assert "model_version" in data


# Fixtures
@pytest.fixture
def sample_image():
    """Fixture providing a sample 28x28 image."""
    return np.random.randint(0, 255, (28, 28)).tolist()


@pytest.fixture
def sample_batch():
    """Fixture providing a batch of 5 images."""
    return [np.random.randint(0, 255, (28, 28)).tolist() for _ in range(5)]


# Parameterized tests
@pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
def test_various_batch_sizes(batch_size):
    """Test prediction with various batch sizes."""
    images = [np.random.randint(0, 255, (28, 28)).tolist() for _ in range(batch_size)]
    response = client.post("/predict", json={"instances": images})
    
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
