#!/usr/bin/env python3
"""
Lecture 83 - Hands-On Lab Script
End-to-end deployment pipeline

Usage:
    python 06_hands_on_lab_deploy_sentiment_or_cnn.py
"""

import sys
import json
import time
import requests
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save_model(models_dir, timestamp):
    """Train and save CNN model."""
    logger.info("Loading Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Use subset
    x_train = x_train[:10000].astype('float32') / 255.0
    x_test = x_test[:2000].astype('float32') / 255.0
    y_train = y_train[:10000]
    y_test = y_test[:2000]
    
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Build model
    logger.info("Building model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    logger.info("Training model...")
    model.fit(x_train, y_train, epochs=3, batch_size=128, 
              validation_split=0.2, verbose=1)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    savedmodel_path = models_dir / f"fashion_cnn_lab_savedmodel_{timestamp}"
    model.save(savedmodel_path, save_format='tf')
    logger.info(f"✓ Model saved: {savedmodel_path}")
    
    # Save config
    config = {
        'normalization': 'divide_by_255',
        'input_shape': [28, 28, 1],
        'num_classes': 10,
        'class_names': [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    }
    
    config_path = models_dir / f"preprocessing_config_{timestamp}.pkl"
    joblib.dump(config, config_path)
    logger.info(f"✓ Config saved: {config_path}")
    
    # Save metadata
    metadata = {
        'model_name': 'fashion_cnn_lab',
        'version': timestamp,
        'test_accuracy': float(test_acc),
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = models_dir / f"model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved: {metadata_path}")
    
    return test_acc


def test_api(api_url="http://localhost:8000"):
    """Test API endpoints."""
    logger.info(f"Testing API at {api_url}...")
    
    # Health check
    try:
        response = requests.get(f"{api_url}/ping", timeout=2)
        if response.status_code == 200:
            logger.info("✓ Health check passed")
        else:
            logger.error("✗ Health check failed")
            return False
    except:
        logger.error("✗ Server not running")
        return False
    
    # Metadata
    response = requests.get(f"{api_url}/metadata")
    if response.status_code == 200:
        logger.info("✓ Metadata endpoint working")
    
    # Prediction
    (_, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    test_image = x_test[0:1].tolist()
    
    response = requests.post(
        f"{api_url}/predict",
        json={"instances": test_image}
    )
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"✓ Prediction: {result['predictions'][0]['class_name']}")
        return True
    else:
        logger.error("✗ Prediction failed")
        return False


def benchmark_api(api_url="http://localhost:8000", num_requests=50):
    """Benchmark API performance."""
    logger.info(f"Benchmarking with {num_requests} requests...")
    
    (_, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    
    latencies = []
    for i in range(num_requests):
        test_image = x_test[i:i+1].tolist()
        
        start = time.time()
        response = requests.post(
            f"{api_url}/predict",
            json={"instances": test_image}
        )
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            latencies.append(latency)
    
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Mean latency: {np.mean(latencies):.2f} ms")
    logger.info(f"  P95 latency: {np.percentile(latencies, 95):.2f} ms")
    logger.info(f"  Throughput: {1000 / np.mean(latencies):.2f} req/sec")


def main():
    print("="*60)
    print("Lecture 83 - Hands-On Lab: End-to-End Deployment")
    print("="*60)
    
    # Setup
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Train and save model
    print("\nStep 1: Training and Saving Model")
    print("-" * 60)
    test_acc = train_and_save_model(models_dir, timestamp)
    
    # Step 2: Test API
    print("\nStep 2: Testing API")
    print("-" * 60)
    print("Make sure FastAPI server is running:")
    print("  cd apps/fastapi_app")
    print("  uvicorn app:app --reload\n")
    
    input("Press Enter when server is ready...")
    
    api_working = test_api()
    
    # Step 3: Benchmark
    if api_working:
        print("\nStep 3: Benchmarking")
        print("-" * 60)
        benchmark_api()
    
    # Summary
    print("\n" + "="*60)
    print("Lab Complete!")
    print("="*60)
    print(f"\nResults:")
    print(f"  Model accuracy: {test_acc:.4f}")
    print(f"  Model timestamp: {timestamp}")
    print(f"  API status: {'✓ Working' if api_working else '✗ Not working'}")
    print("\nNext steps:")
    print("  1. Build Docker image")
    print("  2. Deploy to cloud")
    print("  3. Set up monitoring")


if __name__ == "__main__":
    main()
