# Contributing to Machine Learning Mastery (MLM-101)

First off, thank you for considering contributing to MLM-101! 🎉

The following is a set of guidelines for contributing to this course repository. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Contributing Code](#contributing-code)
  - [Improving Documentation](#improving-documentation)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

---

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. By participating, you are expected to uphold this standard. Please report unacceptable behavior to hello@flowdiary.ai.

**Be respectful, be kind, be collaborative.**

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**How to Submit a Good Bug Report:**

1. **Use a clear and descriptive title**
2. **Describe the exact steps to reproduce the problem**
3. **Provide specific examples** (code snippets, screenshots)
4. **Describe the behavior you observed** and what you expected
5. **Include environment details:**
   - OS (macOS, Windows, Linux)
   - Python version (`python --version`)
   - Package versions (`pip list`)

**Template:**

```markdown
## Bug Description

[Clear description of the bug]

## Steps to Reproduce

1. Go to '...'
2. Run command '...'
3. See error

## Expected Behavior

[What should happen]

## Actual Behavior

[What actually happens]

## Environment

- OS: macOS 13.0
- Python: 3.10.5
- Dependencies: See attached pip list
```

---

### Suggesting Enhancements

We welcome suggestions for new features, projects, or notebooks!

**Before Suggesting:**

- Check if the enhancement has already been suggested
- Consider if it fits the course scope (beginner to advanced ML)

**How to Suggest:**

1. Open a GitHub Issue with the label `enhancement`
2. Provide a clear title and description
3. Explain why this enhancement would be useful
4. Include examples or mockups if applicable

---

### Contributing Code

**Areas to Contribute:**

- 🐛 Bug fixes
- 📓 New Jupyter notebooks
- 🚀 New projects (with datasets)
- 🧪 Unit tests
- 🔧 Utility scripts
- 🌐 Deployment examples

**Process:**

1. **Fork the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/MLM-101.git
   cd MLM-101
   ```

2. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**

   - Follow the [Style Guidelines](#style-guidelines)
   - Add tests if applicable
   - Update documentation

4. **Test your changes**

   ```bash
   # Run notebooks
   jupyter nbconvert --to notebook --execute notebooks/your_notebook.ipynb

   # Run tests
   pytest tests/
   ```

5. **Commit with clear messages**

   ```bash
   git add .
   git commit -m "Add feature: Brief description"
   ```

6. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

---

### Improving Documentation

Documentation improvements are always welcome!

**What to Improve:**

- Typos and grammar
- Clarifications and examples
- Missing setup instructions
- Better explanations in notebooks
- README updates

**Small Changes:**

- Can be submitted directly via GitHub's web interface

**Larger Changes:**

- Follow the [Pull Request Process](#pull-request-process)

---

## Development Setup

### 1. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/MLM-101.git
cd MLM-101
```

### 2. Add Upstream Remote

```bash
git remote add upstream https://github.com/flowdiary/MLM-101.git
```

### 3. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Or using conda
conda create -n mlm101-dev python=3.10
conda activate mlm101-dev
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 nbconvert
```

### 5. Stay Synced with Upstream

```bash
git fetch upstream
git merge upstream/main
```

---

## Pull Request Process

### Before Submitting

✅ **Code Quality:**

- Code runs without errors
- Follows Python style guidelines (PEP 8)
- Includes comments where necessary

✅ **Testing:**

- Notebooks execute successfully
- Scripts run without errors
- Unit tests pass (if applicable)

✅ **Documentation:**

- README updated (if needed)
- Docstrings added to functions
- Comments added to complex code

✅ **Clean Commits:**

- Remove `.DS_Store`, `__pycache__`, `venv/`
- No large binary files (>5MB)
- Commit messages are clear

### PR Guidelines

1. **Title:** Clear and descriptive

   - ✅ "Add sentiment analysis notebook with BERT"
   - ❌ "Update stuff"

2. **Description:** Include:

   - What changes were made
   - Why the changes are needed
   - Related issues (if any)
   - Screenshots (for UI changes)

3. **Review:**
   - Be responsive to feedback
   - Make requested changes promptly
   - Engage in discussion professionally

**Example PR Description:**

```markdown
## Description

Adds a new notebook demonstrating sentiment analysis using BERT transformers.

## Changes

- Created `notebooks/03_nlp/sentiment_analysis_bert.ipynb`
- Added requirements for transformers library
- Updated README with BERT example

## Related Issues

Closes #45

## Screenshots

[If applicable]

## Checklist

- [x] Code runs successfully
- [x] Notebook executes without errors
- [x] Documentation updated
- [x] No large files committed
```

---

## Style Guidelines

### Python Code Style

**Follow PEP 8:**

```python
# Good
def train_model(X_train, y_train, model_type='decision_tree'):
    """
    Train a machine learning model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train

    Returns:
        Trained model object
    """
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Bad
def TrainModel(x,y,t):
    m=DecisionTreeClassifier()
    m.fit(x,y)
    return m
```

**Formatting:**

- Use 4 spaces for indentation (not tabs)
- Max line length: 88 characters (Black formatter)
- Use meaningful variable names

**Run Black formatter:**

```bash
black your_script.py
```

### Jupyter Notebook Style

**Structure:**

1. Title and description (Markdown)
2. Import statements (Code cell)
3. Load data (Code cell with explanation)
4. EDA (Markdown + Code cells)
5. Model training (Code cells with Markdown headers)
6. Evaluation (Code + visualizations)
7. Conclusion (Markdown)

**Best Practices:**

- Clear Markdown headers for sections
- Explain each code cell with comments or Markdown
- Include outputs for all cells
- Restart kernel and run all cells before committing

### Commit Messages

**Format:**

```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**

```bash
feat: Add BERT sentiment analysis notebook

fix: Correct data preprocessing in fraud detection project

docs: Update installation instructions for Windows users

refactor: Reorganize notebooks into topic folders
```

---

## Community

### Getting Help

- 📧 **Email:** hello@flowdiary.ai
- 💬 **Discussions:** [GitHub Discussions](https://github.com/flowdiary/MLM-101/discussions)
- 🐛 **Issues:** [GitHub Issues](https://github.com/flowdiary/MLM-101/issues)

### Recognition

Contributors will be acknowledged in:

- README.md Acknowledgments section
- Release notes (for significant contributions)

---

## Questions?

Don't hesitate to ask! Open a [Discussion](https://github.com/flowdiary/MLM-101/discussions) or email us.

**Thank you for contributing to MLM-101! 🚀**

---

<div align="center">

Made with ❤️ by the MLM-101 community

</div>
