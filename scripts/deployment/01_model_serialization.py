#!/usr/bin/env python3
"""
Lecture 83 - Model Serialization Script

This script demonstrates model serialization for production deployment:
- Training a CNN on Fashion-MNIST
- Saving models in multiple formats (HDF5, SavedModel)
- Serializing preprocessing artifacts
- Loading and validating saved models

Usage:
    python 01_model_serialization.py --output_dir ../models --subset_size 5000
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_cnn_model():
    """Create a simple CNN model for Fashion-MNIST classification."""
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
    return model


def load_and_prepare_data(subset_size=5000, test_size=1000):
    """Load and preprocess Fashion-MNIST dataset."""
    logger.info("Loading Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Use subset for fast training
    x_train_subset = x_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    x_test_subset = x_test[:test_size]
    y_test_subset = y_test[:test_size]
    
    # Normalize and reshape
    x_train_subset = x_train_subset.astype('float32') / 255.0
    x_test_subset = x_test_subset.astype('float32') / 255.0
    x_train_subset = x_train_subset.reshape(-1, 28, 28, 1)
    x_test_subset = x_test_subset.reshape(-1, 28, 28, 1)
    
    logger.info(f"Training samples: {x_train_subset.shape[0]}")
    logger.info(f"Test samples: {x_test_subset.shape[0]}")
    
    return (x_train_subset, y_train_subset), (x_test_subset, y_test_subset)


def train_model(model, x_train, y_train, epochs=2, batch_size=128):
    """Train the model."""
    logger.info("Training model...")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    return history


def save_model_h5(model, output_dir, timestamp):
    """Save model in HDF5 format."""
    h5_path = output_dir / f"fashion_mnist_cnn_{timestamp}.h5"
    model.save(h5_path)
    file_size_kb = h5_path.stat().st_size / 1024
    logger.info(f"Model saved as HDF5: {h5_path} ({file_size_kb:.2f} KB)")
    return h5_path


def save_model_savedmodel(model, output_dir, timestamp):
    """Save model in Keras native format (.keras)."""
    # In Keras 3, use .keras extension for the native format
    savedmodel_path = output_dir / f"fashion_mnist_cnn_{timestamp}.keras"
    model.save(savedmodel_path)
    file_size_kb = savedmodel_path.stat().st_size / 1024
    logger.info(f"Model saved as Keras format: {savedmodel_path} ({file_size_kb:.2f} KB)")
    return savedmodel_path


def save_preprocessing_config(output_dir, timestamp):
    """Save preprocessing configuration."""
    preprocessing_config = {
        'normalization': 'divide_by_255',
        'input_shape': (28, 28, 1),
        'num_classes': 10,
        'class_names': [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    }
    
    config_path = output_dir / f"preprocessing_config_{timestamp}.pkl"
    joblib.dump(preprocessing_config, config_path)
    logger.info(f"Preprocessing config saved: {config_path}")
    return config_path, preprocessing_config


def save_model_metadata(output_dir, timestamp, h5_path, savedmodel_path, 
                        config_path, test_acc, subset_size):
    """Save model metadata for tracking."""
    model_metadata = {
        'model_name': 'fashion_mnist_cnn',
        'version': timestamp,
        'framework': 'tensorflow',
        'framework_version': tf.__version__,
        'model_type': 'cnn',
        'dataset': 'fashion_mnist',
        'training_samples': subset_size,
        'test_accuracy': float(test_acc),
        'input_shape': [28, 28, 1],
        'output_classes': 10,
        'saved_formats': ['h5', 'savedmodel'],
        'created_at': datetime.now().isoformat(),
        'files': {
            'h5_model': str(h5_path.name),
            'savedmodel': str(savedmodel_path.name),
            'preprocessing': str(config_path.name)
        }
    }
    
    metadata_path = output_dir / f"model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"Model metadata saved: {metadata_path}")
    return metadata_path


def test_model_loading(h5_path, savedmodel_path, x_test, y_test):
    """Test that saved models can be loaded and produce consistent results."""
    logger.info("Testing model loading and consistency...")
    
    # Load both models
    loaded_h5 = tf.keras.models.load_model(h5_path)
    loaded_savedmodel = tf.keras.models.load_model(savedmodel_path)
    
    # Test predictions
    h5_preds = loaded_h5.predict(x_test[:5], verbose=0)
    savedmodel_preds = loaded_savedmodel.predict(x_test[:5], verbose=0)
    
    # Assert predictions match
    np.testing.assert_allclose(h5_preds, savedmodel_preds, rtol=1e-5)
    logger.info("✓ Both models produce identical predictions")
    
    # Test accuracy
    _, h5_acc = loaded_h5.evaluate(x_test, y_test, verbose=0)
    logger.info(f"✓ Loaded HDF5 model accuracy: {h5_acc:.4f}")
    
    return True


def preprocess_image(image_array, config):
    """Apply preprocessing based on saved config."""
    if config['normalization'] == 'divide_by_255':
        image_array = image_array.astype('float32') / 255.0
    
    if len(image_array.shape) == 2:
        image_array = image_array.reshape(1, 28, 28, 1)
    elif len(image_array.shape) == 3:
        image_array = image_array.reshape(-1, 28, 28, 1)
    
    return image_array


def predict_with_labels(model, image_array, config):
    """Complete inference pipeline with class labels."""
    processed = preprocess_image(image_array, config)
    predictions = model.predict(processed, verbose=0)
    predicted_classes = predictions.argmax(axis=1)
    predicted_labels = [config['class_names'][i] for i in predicted_classes]
    
    return predicted_labels, predictions


def main():
    parser = argparse.ArgumentParser(description='Model Serialization Demo')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directory to save models')
    parser.add_argument('--subset_size', type=int, default=5000,
                        help='Training subset size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    
    args = parser.parse_args()
    
    # Setup
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data(args.subset_size)
    
    # Create and train model
    model = create_cnn_model()
    logger.info(f"Model has {model.count_params():,} parameters")
    
    history = train_model(model, x_train, y_train, args.epochs, args.batch_size)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    # Save models
    h5_path = save_model_h5(model, output_dir, timestamp)
    savedmodel_path = save_model_savedmodel(model, output_dir, timestamp)
    config_path, preprocessing_config = save_preprocessing_config(output_dir, timestamp)
    metadata_path = save_model_metadata(
        output_dir, timestamp, h5_path, savedmodel_path,
        config_path, test_acc, args.subset_size
    )
    
    # Test loading
    test_model_loading(h5_path, savedmodel_path, x_test, y_test)
    
    # Demo prediction pipeline
    loaded_model = tf.keras.models.load_model(savedmodel_path)
    loaded_config = joblib.load(config_path)
    
    # Get raw test image (not normalized)
    (_, _), (x_test_raw, y_test_raw) = tf.keras.datasets.fashion_mnist.load_data()
    test_image = x_test_raw[0]
    
    labels, probs = predict_with_labels(loaded_model, test_image, loaded_config)
    logger.info(f"Sample prediction: {labels[0]} (confidence: {probs[0].max():.4f})")
    logger.info(f"Actual class: {loaded_config['class_names'][y_test_raw[0]]}")
    
    logger.info("\n" + "="*60)
    logger.info("Model serialization complete!")
    logger.info(f"Saved files:")
    logger.info(f"  - HDF5 model: {h5_path.name}")
    logger.info(f"  - SavedModel: {savedmodel_path.name}")
    logger.info(f"  - Preprocessing: {config_path.name}")
    logger.info(f"  - Metadata: {metadata_path.name}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
