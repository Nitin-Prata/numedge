# NumEdge

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**A lightweight, safety-first machine learning library built on NumPy**

*Making classical ML transparent, safe, and beginner-friendly*

[Installation](#-installation) • [Quick Start](#-quick-start) • [Features](#-what-makes-numedge-different) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🎯 Why NumEdge?

Most machine learning libraries are either too simple for real work or too complex to understand. NumEdge bridges this gap.

**NumEdge is built for:**
- 🎓 **Students & Learners** — Every algorithm is written in pure NumPy/Python. No hidden complexity, no black boxes.
- 🛡️ **Safety-Conscious Developers** — Built-in warnings catch common ML mistakes before they become production bugs.
- 📊 **Tabular Data Practitioners** — First-class DataFrame support with automatic preprocessing.
- 🔬 **Research & Education** — Clean, readable source code that teaches as much as it performs.

> **Philosophy:** Readable over fast. Understanding over optimization. Safety over convenience.

---

## ✨ What Makes NumEdge Different?

### 1️⃣ **Transparent by Design**
Every algorithm implemented in pure NumPy and Python. Open the source, understand the math, learn how ML actually works.

### 2️⃣ **Safety-First Approach**
NumEdge actively protects you from common mistakes:
- ⚠️ Warns when evaluating on training data
- ⚠️ Detects missing `random_state` for reproducibility
- ⚠️ Catches shape mismatches and data leakage patterns
- ⚠️ Validates preprocessing pipelines

### 3️⃣ **Tabular-First Design**
Real-world data comes in CSVs with mixed types. NumEdge handles this automatically:

```python
from numedge.tabular import TabularClassifier
from numedge.models.ensemble import RandomForestClassifier

# Just pass your DataFrame — NumEdge handles the rest
model = TabularClassifier(
    estimator=RandomForestClassifier(),
    target="label"
)

model.fit(df_train)  # Auto-detects numeric/categorical columns
predictions = model.predict(df_test)  # Auto-preprocesses test data
```

### 4️⃣ **Built-in Hyperparameter Intelligence**
Every model knows its own optimal search spaces:

```python
model = RandomForestClassifier()
search_space = model.get_search_space()  # Recommended ranges for tuning
# {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None], ...}
```

### 5️⃣ **Streaming-Friendly**
Supports incremental learning where it makes sense:

```python
model = LinearRegression()
for batch in data_stream:
    model.partial_fit(X_batch, y_batch)  # Learn incrementally
```

---

## 🚀 Installation

### Requirements
- Python 3.8 or higher
- NumPy (required)
- pandas (optional, for tabular features)

### Install from Source

```bash
git clone https://github.com/Nitin-Prata/numedge.git
cd numedge
pip install -e .
```

> **Coming soon to PyPI:** `pip install numedge`

---

## 🔧 Quick Start

### Basic Regression

```python
from numedge.models.linear_models import LinearRegression
from numedge.model_selection import train_test_split

# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")
```

### Classification with Safety Checks

```python
from numedge.models.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# NumEdge warns you if you accidentally evaluate on training data
train_score = model.score(X_train, y_train)  # ⚠️ Warning: Evaluating on training data!
test_score = model.score(X_test, y_test)     # ✓ Correct evaluation
```

### Tabular Data with DataFrames

```python
from numedge.tabular import TabularRegressor
from numedge.models.ensemble import GradientBoostingRegressor

# Your DataFrame has mixed types: numeric, categorical, dates, etc.
model = TabularRegressor(
    estimator=GradientBoostingRegressor(),
    target="price"
)

# NumEdge automatically:
# - Detects column types
# - Scales numeric features
# - Encodes categorical features
# - Handles missing values
model.fit(df_train)

predictions = model.predict(df_test)
```

---

## 🧠 Algorithms

### Supervised Learning

**Linear Models**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Logistic Regression

**Tree-Based Models**
- Decision Trees (Classifier & Regressor)
- Random Forest (Classifier & Regressor)
- Extra Trees
- Gradient Boosting
- AdaBoost
- Bagging

**Support Vector Machines**
- SVM Classifier
- SVM Regressor

**Neighbors**
- KNN Classifier
- KNN Regressor

**Naive Bayes**
- Gaussian Naive Bayes
- Multinomial Naive Bayes

**Boosting**
- NumPy-based XGBoost implementation

### Unsupervised Learning

**Clustering**
- K-Means
- DBSCAN
- Hierarchical Clustering

**Dimensionality Reduction**
- PCA (Principal Component Analysis)

### Preprocessing
- Standard Scaler
- Min-Max Scaler
- One-Hot Encoder
- Label Encoder

### Model Selection
- Train-Test Split
- Cross-Validation
- Hyperparameter Search

---

## 📁 Project Structure

```
numedge/
├── src/numedge/
│   ├── core/              # BaseEstimator, mixins, optimizers, exceptions
│   ├── models/            # All supervised learning algorithms
│   │   ├── linear_models/
│   │   ├── ensemble/
│   │   ├── tree/
│   │   ├── svm/
│   │   └── neighbors/
│   ├── cluster/           # Clustering algorithms
│   ├── decomposition/     # PCA and dimensionality reduction
│   ├── preprocessing/     # Scalers, encoders, transformers
│   ├── model_selection/   # Cross-validation, search utilities
│   ├── metrics/           # Evaluation metrics
│   ├── tabular/           # DataFrame-friendly helpers
│   └── utils/             # Internal utilities and validators
├── tests/                 # Comprehensive unit tests
├── examples/              # Usage examples and tutorials
└── docs/                  # Documentation (coming soon)
```

---

## 🎓 Learning Resources

NumEdge is designed to be educational. Each algorithm includes:
- 📖 Clear docstrings explaining the math
- 🔍 Readable source code with extensive comments
- ✅ Type hints for better IDE support
- 🧪 Unit tests showing expected behavior

**Recommended Learning Path:**
1. Start with `LinearRegression` — see how gradient descent works
2. Move to `DecisionTree` — understand recursive splitting
3. Explore `RandomForest` — see how ensemble methods combine weak learners
4. Deep dive into `GradientBoosting` — learn advanced boosting

---

## 📚 Documentation

Full documentation is in development and will cover:
- 📘 API Reference
- 🏗️ Architecture & Design Decisions
- 📝 Tutorials & Examples
- 🧑‍🏫 Algorithm Explanations
- 🤝 Contribution Guide

---

## 🤝 Contributing

NumEdge is actively developed and welcomes contributions!

**Ways to contribute:**
- 🐛 Report bugs and issues
- 💡 Suggest new features or algorithms
- 📖 Improve documentation
- ✨ Submit pull requests
- ⭐ Star the repository

Please check our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

---

## 🛣️ Roadmap

- [ ] Complete test coverage (>90%)
- [ ] Comprehensive documentation site
- [ ] PyPI release
- [ ] Additional algorithms (LightGBM-style, CatBoost-style)
- [ ] Performance benchmarks
- [ ] Interactive tutorials and notebooks
- [ ] CI/CD pipeline
- [ ] GPU acceleration (optional)

---

## 📄 License

NumEdge is released under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Nitin Pratap Singh**

- GitHub: [@Nitin-Prata](https://github.com/Nitin-Prata)
- LinkedIn: [Nitin Singh](https://www.linkedin.com/in/nitin-singh-bb7907298/)
- X (Twitter): [@prata42085](https://x.com/prata42085)
- Email: nitinpratap997@gmail.com

---

## 🙏 Acknowledgments

NumEdge is inspired by the need for transparent, educational machine learning tools. While it shares API conventions with popular libraries, it maintains its own identity focused on clarity, safety, and learning.

Special thanks to the open-source ML community for creating an ecosystem that makes projects like this possible.

---

## ⭐ Star History

If you find NumEdge useful, please consider starring the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=Nitin-Prata/numedge&type=Date)](https://star-history.com/#Nitin-Prata/numedge&Date)

---

<div align="center">

**Made with ❤️ for the ML community**

[Report Bug](https://github.com/Nitin-Prata/numedge/issues) • [Request Feature](https://github.com/Nitin-Prata/numedge/issues) • [Ask Question](https://github.com/Nitin-Prata/numedge/discussions)

</div>