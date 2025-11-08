# Trained Models

This directory stores trained machine learning models.

## Model Storage

Trained models are saved here after running project scripts or notebooks. Common formats include:

- **Scikit-learn models**: `.pkl`, `.joblib`
- **TensorFlow/Keras models**: `.h5`, `.keras`, SavedModel format
- **PyTorch models**: `.pth`, `.pt`

## Model Organization

Organize models by project:

```
models/
├── sales_forecasting/
│   ├── decision_tree_v1.joblib
│   └── random_forest_v2.joblib
├── fraud_detection/
│   └── classifier_model.pkl
├── course_recommendation/
│   └── decision_tree_model.joblib
├── sentiment_analysis/
│   └── bert_sentiment.h5
└── image_classification/
    └── cnn_model.h5
```

## Saved Model Files

After training, models are typically saved in the following locations:

- **Sales Forecasting**: `projects/01_sales_forecasting/models/`
- **Fraud Detection**: `projects/02_fraud_detection/models/`
- **Course Recommendation**: `projects/03_course_recommendation/models/`

## Loading Models

### Scikit-learn (joblib)

```python
import joblib
model = joblib.load('models/sales_forecasting/model.joblib')
```

### TensorFlow/Keras

```python
from tensorflow import keras
model = keras.models.load_model('models/cnn_model.h5')
```

### PyTorch

```python
import torch
model = torch.load('models/model.pth')
```

## Model Versioning

Use descriptive names with versions:

- `sales_dt_v1.joblib` (Decision Tree, version 1)
- `fraud_rf_v2.pkl` (Random Forest, version 2)
- `sentiment_bert_20250108.h5` (BERT model, date-based)

## Git LFS (Large File Storage)

For models >100MB, consider using Git LFS:

```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.pkl"
git add .gitattributes
```

## Best Practices

1. ✅ **Name descriptively**: Include algorithm and version
2. ✅ **Document performance**: Keep metrics in model README
3. ✅ **Version control**: Track model changes
4. ✅ **Compress large models**: Use appropriate formats
5. ✅ **Exclude from Git**: Add to `.gitignore` if too large

## Model Documentation Template

For each model, document:

- **Algorithm**: Decision Tree, CNN, BERT, etc.
- **Training date**: When was it trained?
- **Dataset**: What data was used?
- **Performance**: Accuracy, R², F1-score, etc.
- **Hyperparameters**: Key settings used
- **Usage**: How to load and use the model

## Need Help?

- See project README files for model-specific documentation
- Contact: support@flowdiary.com.ng
