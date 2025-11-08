# Datasets

This directory stores external datasets that are not included in the repository due to size constraints.

## Included Datasets

All project-specific datasets are stored in their respective project directories:

- **Sales Data**: `projects/01_sales_forecasting/data/sales_data.csv`
- **Fraud Data**: `projects/02_fraud_detection/data/fraud_data.csv`
- **Course Data**: `projects/03_course_recommendation/data/course_data.csv`

## External Datasets

For deep learning and advanced exercises, you may need to download additional datasets:

### Image Datasets

- **CIFAR-10**: Automatically downloaded by TensorFlow/Keras
- **MNIST**: Automatically downloaded by TensorFlow/Keras
- **ImageNet subset**: See instructions in relevant notebooks

### NLP Datasets

- **IMDB Reviews**: Automatically downloaded by TensorFlow/Keras
- **Custom text corpora**: See RAG notebooks for PDF sources

### Download Instructions

Some notebooks include automatic dataset downloading. For manual downloads:

```bash
# Navigate to scripts directory
cd scripts/data

# Run download script
python download_data.py
```

## Data Storage Guidelines

1. **Small datasets** (<10MB): Include in project `data/` folders
2. **Medium datasets** (10-100MB): Download via script, store here
3. **Large datasets** (>100MB): Provide download links, don't commit to Git

## Dataset Organization

When adding new datasets:

```
data/
├── images/           # Image datasets
├── text/             # Text corpora
├── tabular/          # CSV/Excel files
└── processed/        # Preprocessed datasets
```

## Data Privacy

⚠️ **Important**: Never commit sensitive or private data to the repository.

- Use `.gitignore` to exclude large files
- Anonymize personal information
- Follow data protection regulations (GDPR, CCPA)

## Need Help?

- See: [Dataset Guide](../docs/guides/dataset-guide.md)
- Contact: support@flowdiary.com.ng
