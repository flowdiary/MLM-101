#!/usr/bin/env python3
"""
Download sample data for Lecture 83 deployment notebooks.
"""

import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import json

def download_fashion_mnist():
    """Download Fashion-MNIST dataset."""
    print("Downloading Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Save a small subset for quick testing
    data_dir = Path(__file__).parent
    
    # Save test samples
    np.save(data_dir / 'x_test_sample.npy', x_test[:100])
    np.save(data_dir / 'y_test_sample.npy', y_test[:100])
    
    print(f"✓ Saved 100 test samples to {data_dir}")
    print(f"  Shape: {x_test[:100].shape}")
    
    return True

def create_sample_documents():
    """Create sample documents for RAG demo."""
    documents = [
        {
            "id": 1,
            "title": "Introduction to Deep Learning",
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has revolutionized computer vision, natural language processing, and many other fields. Modern architectures like transformers and CNNs have achieved human-level performance on many tasks."
        },
        {
            "id": 2,
            "title": "Model Deployment Best Practices",
            "content": "Model deployment involves taking a trained machine learning model and making it available for predictions in a production environment. This includes serialization, API creation, monitoring, and versioning. Key considerations include latency, throughput, and reliability."
        },
        {
            "id": 3,
            "title": "FastAPI for ML Services",
            "content": "FastAPI is a modern, fast web framework for building APIs with Python. It's ideal for serving machine learning models due to its high performance and automatic data validation. FastAPI includes features like async support, dependency injection, and automatic OpenAPI documentation."
        },
        {
            "id": 4,
            "title": "Docker and Containerization",
            "content": "Docker containers package applications and their dependencies together, ensuring consistent behavior across different environments. They're essential for deploying ML models reliably. Containers provide isolation, portability, and scalability for production systems."
        },
        {
            "id": 5,
            "title": "RAG Systems Explained",
            "content": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context for generating accurate responses. This approach reduces hallucinations and grounds responses in factual information."
        },
        {
            "id": 6,
            "title": "TensorFlow vs PyTorch",
            "content": "TensorFlow and PyTorch are the two most popular deep learning frameworks. TensorFlow offers production-ready tools like TensorFlow Serving and TensorFlow Lite, while PyTorch is favored for research due to its dynamic computation graphs and intuitive API."
        }
    ]
    
    data_dir = Path(__file__).parent
    with open(data_dir / 'sample_documents.json', 'w') as f:
        json.dump(documents, f, indent=2)
    
    print(f"✓ Created {len(documents)} sample documents")
    return documents

def main():
    print("=" * 60)
    print("Lecture 83 - Data Download Script")
    print("=" * 60)
    
    # Download Fashion-MNIST
    download_fashion_mnist()
    
    # Create sample documents
    create_sample_documents()
    
    print("\n" + "=" * 60)
    print("✓ All data ready!")
    print("=" * 60)

if __name__ == "__main__":
    main()
