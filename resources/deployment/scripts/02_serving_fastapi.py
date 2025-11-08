#!/usr/bin/env python3
"""
Lecture 83 - FastAPI Serving Script
Demonstrates API testing and performance benchmarking.

Usage:
    python 02_serving_fastapi.py --api_url http://localhost:8000
"""

import argparse
import requests
import numpy as np
import time
import json
import sys

def check_server_health(api_url):
    """Check if the API server is running."""
    try:
        response = requests.get(f"{api_url}/ping", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_metadata(api_url):
    """Get model metadata from API."""
    response = requests.get(f"{api_url}/metadata")
    if response.status_code == 200:
        return response.json()
    return None

def make_prediction(api_url, images):
    """Make predictions via API."""
    payload = {"instances": images}
    response = requests.post(
        f"{api_url}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    return response

def benchmark_api(api_url, num_requests=50):
    """Benchmark API performance."""
    import tensorflow as tf
    
    # Load test images
    (_, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    
    latencies = []
    
    print(f"Running {num_requests} requests...")
    for i in range(num_requests):
        test_image = x_test[i:i+1].tolist()
        
        start = time.time()
        response = make_prediction(api_url, test_image)
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            latencies.append(latency)
        else:
            print(f"Request {i+1} failed: {response.status_code}")
    
    # Print statistics
    if latencies:
        print(f"\n{'='*60}")
        print("Performance Metrics:")
        print(f"  Total requests: {len(latencies)}")
        print(f"  Mean latency: {np.mean(latencies):.2f} ms")
        print(f"  Median latency: {np.median(latencies):.2f} ms")
        print(f"  P95 latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Min latency: {np.min(latencies):.2f} ms")
        print(f"  Max latency: {np.max(latencies):.2f} ms")
        print(f"  Throughput: {1000 / np.mean(latencies):.2f} req/sec")
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='FastAPI Server Testing')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                        help='API server URL')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--num_requests', type=int, default=50,
                        help='Number of requests for benchmark')
    
    args = parser.parse_args()
    
    # Check server health
    print("Checking server status...")
    if not check_server_health(args.api_url):
        print(f"✗ Server not running at {args.api_url}")
        print("\nStart server with:")
        print("  cd apps/fastapi_app")
        print("  uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)
    
    print(f"✓ Server is running at {args.api_url}\n")
    
    # Get metadata
    print("Fetching model metadata...")
    metadata = get_metadata(args.api_url)
    if metadata:
        print(json.dumps(metadata, indent=2))
        print()
    
    # Make sample prediction
    print("Making sample prediction...")
    import tensorflow as tf
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    test_images = x_test[:3].tolist()
    response = make_prediction(args.api_url, test_images)
    
    if response.status_code == 200:
        results = response.json()
        print("\nPredictions:")
        for i, pred in enumerate(results['predictions']):
            print(f"  Image {i+1}: {pred['class_name']} (confidence: {pred['confidence']:.4f})")
        print()
    else:
        print(f"Prediction failed: {response.status_code}")
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_api(args.api_url, args.num_requests)

if __name__ == "__main__":
    main()
