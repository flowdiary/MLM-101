# Troubleshooting Guide

See main [README.md](../../README.md#-troubleshooting) for common issues and solutions.

## Common Issues

### 1. ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'xyz'`

**Solution:**

```bash
# Ensure you're in the correct environment
source venv/bin/activate  # or: conda activate mlm101

# Install missing package
pip install xyz
```

### 2. Jupyter Kernel Not Found

**Problem:** Kernel not available in Jupyter

**Solution:**

```bash
# Install IPython kernel
python -m ipykernel install --user --name=mlm101
```

### 3. Permission Denied (macOS/Linux)

**Problem:** Permission errors during installation

**Solution:**

```bash
# Use pip with --user flag
pip install --user -r requirements.txt
```

### 4. Port Already in Use

**Problem:** Streamlit or FastAPI can't start

**Solution:**

```bash
# Change port for Streamlit
streamlit run app.py --server.port 8502

# Change port for FastAPI
uvicorn app:app --port 8001
```

### 5. CUDA/GPU Issues

**Problem:** TensorFlow can't find GPU

**Solution:**

```bash
# Verify TensorFlow GPU installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CPU-only version if no GPU
pip install tensorflow-cpu
```

## Additional Resources

- [GitHub Issues](https://github.com/flowdiary/MLM-101/issues)
- Email: support@flowdiary.com.ng
- Main README: [README.md](../../README.md)
