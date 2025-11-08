#!/usr/bin/env python3
"""
Lecture 83 - Real-Time Inference Script

Usage:
    python 05_real_time_inference.py
"""

import asyncio
import time
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch predictions for higher throughput."""
    
    def __init__(self, model=None, max_batch_size=32, max_wait_time=0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = []
        self.lock = asyncio.Lock()
    
    async def predict(self, data):
        """Add to batch and wait for result."""
        future = asyncio.Future()
        
        async with self.lock:
            self.queue.append((data, future))
        
        # Trigger batch if full
        if len(self.queue) >= self.max_batch_size:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated batch."""
        async with self.lock:
            if not self.queue:
                return
            
            batch = self.queue.copy()
            self.queue.clear()
        
        # Extract data
        data_batch = [d for d, _ in batch]
        futures = [f for _, f in batch]
        
        # Simulate batch prediction
        await asyncio.sleep(0.01)  # Mock inference time
        results = [np.random.random() for _ in data_batch]
        
        # Set results
        for future, result in zip(futures, results):
            future.set_result(result)
        
        logger.info(f"Processed batch of {len(batch)} items")


async def demo_batch_inference():
    """Demo batch inference."""
    predictor = BatchPredictor(max_batch_size=8)
    
    # Send 20 requests
    tasks = []
    for i in range(20):
        task = predictor.predict({"id": i, "data": np.random.random((28, 28))})
        tasks.append(task)
        await asyncio.sleep(0.01)  # Small delay
    
    # Wait for all results
    results = await asyncio.gather(*tasks)
    
    logger.info(f"Received {len(results)} results")
    return results


def measure_latency():
    """Measure inference latency."""
    # Simulate predictions
    latencies = []
    
    for _ in range(100):
        start = time.time()
        # Simulate inference
        time.sleep(0.01)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    print("\nLatency Statistics:")
    print(f"  Mean: {np.mean(latencies):.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95: {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99: {np.percentile(latencies, 99):.2f} ms")


def main():
    print("="*60)
    print("Real-Time Inference Demo")
    print("="*60)
    
    # Demo 1: Latency measurement
    print("\n1. Measuring Latency...")
    measure_latency()
    
    # Demo 2: Batch inference
    print("\n2. Batch Inference Demo...")
    asyncio.run(demo_batch_inference())
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
