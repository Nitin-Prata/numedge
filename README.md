<div align="center">

# 🎯 NumEdge

### *Machine Learning That Makes Sense*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

**A lightweight, intelligent machine learning library built on pure NumPy**

*Where transparency meets power, and learning meets doing*

---

### [📦 Install](#-installation) • [🚀 Quick Start](#-quick-start) • [✨ Features](#-core-philosophy) • [🤝 Contribute](#-contributing)

---

</div>

<br>

## 🌟 Why NumEdge Exists

Machine learning shouldn't feel like magic. It should be **transparent**, **intuitive**, and **intelligent**.

NumEdge was born from a simple belief: **great ML tools should teach you while you build**. Every algorithm is implemented in pure NumPy and Python—no hidden layers, no cryptic C extensions, just clean, readable code that helps you understand what's really happening under the hood.

<div align="center">

### 🎓 Built For

| Students & Learners | Data Scientists | Researchers | Educators |
|:---:|:---:|:---:|:---:|
| Pure Python code you can actually read | Tabular-first with DataFrame support | Reproducible & well-documented | Perfect teaching tool |

</div>

<br>

---

<br>

## ✨ Core Philosophy

<table>
<tr>
<td width="33%" align="center">

### 🔍 **Transparent by Design**

Every algorithm written in pure NumPy/Python. Open any file and understand exactly how the math works. No black boxes, no magic.

</td>
<td width="33%" align="center">

### 🛡️ **Intelligent Warnings**

Built-in safeguards catch common mistakes before they become bugs. Data leakage? Wrong evaluation? Missing random state? We've got you covered.

</td>
<td width="33%" align="center">

### 📊 **Tabular-First**

Real data comes in CSVs and DataFrames. NumEdge handles mixed types, preprocessing, and encoding automatically—no pipelines required.

</td>
</tr>
</table>

<br>

<div align="center">

### 🎯 *Readable over Fast • Understanding over Optimization • Clarity over Complexity*

</div>

<br>

---

<br>

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Nitin-Prata/numedge.git
cd numedge

# Install in development mode
pip install -e .
```

> 📦 **Coming Soon:** `pip install numedge`

**Requirements:**
- Python 3.8 or higher
- NumPy (core dependency)
- pandas (optional, for tabular features)

<br>

---

<br>

## 💡 Examples

### 🔹 Linear Regression

```python
from numedge.models.linear_models import LinearRegression
from numedge.model_selection import train_test_split
import numpy as np

# Generate sample data
X = np.random.randn(1000, 5)
y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(1000) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
print(f"Training R²: {model.score(X_train, y_train):.4f}")
print(f"Testing R²: {model.score(X_test, y_test):.4f}")

# Make predictions
predictions = model.predict(X_test)
```

<br>

### 🔹 Random Forest with Hyperparameter Search

```python
from numedge.models.ensemble import RandomForestClassifier
from numedge.model_selection import GridSearchCV

# Create model
rf = RandomForestClassifier(random_state=42)

# Get recommended hyperparameter search space
search_space = rf.get_search_space()
print(f"Recommended search space: {search_space}")

# Perform grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=search_space,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

<br>

### 🔹 Tabular Data (DataFrames)

```python
import pandas as pd
from numedge.tabular import TabularClassifier
from numedge.models.ensemble import GradientBoostingClassifier

# Your real-world DataFrame with mixed types
df = pd.DataFrame({
    'age': [25, 35, 45, 22, 55],
    'income': [50000, 75000, 90000, 45000, 120000],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'purchased': [0, 1, 1, 0, 1]
})

# Create tabular classifier
model = TabularClassifier(
    estimator=GradientBoostingClassifier(random_state=42),
    target='purchased'
)

# NumEdge automatically:
# ✅ Detects numeric vs categorical columns
# ✅ Scales numeric features
# ✅ One-hot encodes categorical features
# ✅ Handles train/test consistency

model.fit(df)
predictions = model.predict(df)
```

<br>

### 🔹 K-Means Clustering

```python
from numedge.cluster import KMeans
import matplotlib.pyplot as plt

# Create clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.predict(X)
centers = kmeans.cluster_centers_

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolors='black')
plt.title('K-Means Clustering')
plt.show()
```

<br>

### 🔹 Incremental Learning (Streaming)

```python
from numedge.models.linear_models import SGDRegressor

# Initialize model
model = SGDRegressor(learning_rate=0.01)

# Learn from data in batches (useful for large datasets)
for batch_X, batch_y in data_stream:
    model.partial_fit(batch_X, batch_y)

# Final predictions
final_predictions = model.predict(X_test)
```

<br>

---

<br>

## 🧠 Available Algorithms

<details>
<summary><b>📈 Supervised Learning</b></summary>

<br>

**Linear Models**
- `LinearRegression` — Ordinary least squares
- `Ridge` — L2 regularized regression
- `Lasso` — L1 regularized regression
- `ElasticNet` — Combined L1 + L2 regularization
- `LogisticRegression` — Binary & multiclass classification
- `SGDRegressor` — Stochastic gradient descent regression
- `SGDClassifier` — Stochastic gradient descent classification

**Tree-Based Models**
- `DecisionTreeClassifier` — CART algorithm
- `DecisionTreeRegressor` — Regression trees
- `RandomForestClassifier` — Ensemble of decision trees
- `RandomForestRegressor` — Ensemble for regression
- `ExtraTreesClassifier` — Extremely randomized trees
- `ExtraTreesRegressor` — Extra trees for regression

**Ensemble Methods**
- `GradientBoostingClassifier` — Gradient boosting
- `GradientBoostingRegressor` — Boosting for regression
- `AdaBoostClassifier` — Adaptive boosting
- `AdaBoostRegressor` — AdaBoost for regression
- `BaggingClassifier` — Bootstrap aggregating
- `BaggingRegressor` — Bagging for regression

**Support Vector Machines**
- `SVC` — Support vector classification
- `SVR` — Support vector regression

**Neighbors**
- `KNeighborsClassifier` — K-nearest neighbors classification
- `KNeighborsRegressor` — K-nearest neighbors regression

**Naive Bayes**
- `GaussianNB` — Gaussian Naive Bayes
- `MultinomialNB` — Multinomial Naive Bayes

**Advanced Boosting**
- `XGBClassifier` — NumPy-based XGBoost implementation
- `XGBRegressor` — XGBoost for regression

</details>

<details>
<summary><b>🔍 Unsupervised Learning</b></summary>

<br>

**Clustering**
- `KMeans` — K-means clustering
- `DBSCAN` — Density-based clustering
- `AgglomerativeClustering` — Hierarchical clustering

**Dimensionality Reduction**
- `PCA` — Principal component analysis

</details>

<details>
<summary><b>⚙️ Preprocessing & Utilities</b></summary>

<br>

**Scalers**
- `StandardScaler` — Standardize features (zero mean, unit variance)
- `MinMaxScaler` — Scale features to a range
- `RobustScaler` — Scale using median and IQR

**Encoders**
- `OneHotEncoder` — One-hot encode categorical features
- `LabelEncoder` — Encode labels as integers

**Model Selection**
- `train_test_split` — Split data into train/test sets
- `cross_val_score` — K-fold cross-validation
- `GridSearchCV` — Exhaustive hyperparameter search
- `RandomizedSearchCV` — Randomized hyperparameter search

**Metrics**
- Classification: `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`
- Regression: `r2_score`, `mse`, `mae`, `rmse`

</details>

<br>

---

<br>

## 📁 Project Structure

```
numedge/
│
├── 📂 src/numedge/
│   │
│   ├── 🎯 core/                    # Base classes, mixins, optimizers
│   │   ├── base.py                 # BaseEstimator
│   │   ├── mixins.py               # ClassifierMixin, RegressorMixin
│   │   └── optimizers.py           # Gradient descent variants
│   │
│   ├── 🤖 models/                  # All supervised algorithms
│   │   ├── linear_models/          # Linear regression, Ridge, Lasso, etc.
│   │   ├── ensemble/               # Random Forest, Boosting, Bagging
│   │   ├── tree/                   # Decision Trees
│   │   ├── svm/                    # Support Vector Machines
│   │   ├── neighbors/              # K-Nearest Neighbors
│   │   └── naive_bayes/            # Naive Bayes variants
│   │
│   ├── 🔍 cluster/                 # Clustering algorithms
│   │   ├── kmeans.py
│   │   ├── dbscan.py
│   │   └── hierarchical.py
│   │
│   ├── 📊 decomposition/           # Dimensionality reduction
│   │   └── pca.py
│   │
│   ├── ⚙️ preprocessing/           # Data transformers
│   │   ├── scalers.py
│   │   └── encoders.py
│   │
│   ├── 🎲 model_selection/         # CV, search, split utilities
│   │   ├── split.py
│   │   ├── cross_validation.py
│   │   └── search.py
│   │
│   ├── 📏 metrics/                 # Evaluation metrics
│   │   ├── classification.py
│   │   └── regression.py
│   │
│   ├── 📋 tabular/                 # DataFrame helpers
│   │   ├── classifier.py
│   │   └── regressor.py
│   │
│   └── 🛠️ utils/                   # Internal utilities
│       ├── validation.py
│       ├── checks.py
│       └── warnings.py
│
├── 🧪 tests/                       # Comprehensive test suite
├── 📚 examples/                    # Jupyter notebooks & tutorials
├── 📖 docs/                        # Documentation (coming soon)
└── 📄 README.md                    # You are here!
```

<br>

---

<br>

## 🗺️ Roadmap

<table>
<tr>
<td width="50%">

### ✅ Current Focus
- [x] Core algorithms implementation
- [x] Tabular data support
- [x] Hyperparameter search spaces
- [x] Intelligent warning system
- [ ] Complete test coverage (>90%)
- [ ] Performance benchmarks

</td>
<td width="50%">

### 🔮 Coming Soon
- [ ] Full documentation site
- [ ] PyPI release
- [ ] Interactive tutorials
- [ ] More ensemble methods
- [ ] Advanced feature engineering
- [ ] CI/CD pipeline

</td>
</tr>
</table>

<br>

---

<br>

## 🤝 Contributing

NumEdge is **actively developed** and we'd love your help!

### 🌟 Ways to Contribute

- 🐛 **Report Bugs** — Found an issue? [Open an issue](https://github.com/Nitin-Prata/numedge/issues)
- 💡 **Suggest Features** — Have ideas? [Start a discussion](https://github.com/Nitin-Prata/numedge/discussions)
- 📖 **Improve Docs** — Better explanations, examples, tutorials
- ✨ **Submit Code** — New algorithms, optimizations, fixes
- ⭐ **Star the Repo** — Show your support!

Before contributing code, please read our [**Contributing Guidelines**](CONTRIBUTING.md).

<br>

---

<br>

## 📄 License

NumEdge is open-source software licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for full details.

<br>

---

<br>

## 👨‍💻 Creator

<div align="center">

### **Nitin Pratap Singh**

[![GitHub](https://img.shields.io/badge/GitHub-@Nitin--Prata-181717?style=for-the-badge&logo=github)](https://github.com/Nitin-Prata)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nitin%20Singh-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/nitin-singh-bb7907298/)
[![X](https://img.shields.io/badge/X-@prata42085-000000?style=for-the-badge&logo=x)](https://x.com/prata42085)
[![Email](https://img.shields.io/badge/Email-nitinpratap997@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nitinpratap997@gmail.com)

</div>

<br>

---

<br>

## 🙏 Acknowledgments

NumEdge stands on the shoulders of giants. Inspired by the open-source ML community and driven by a passion for transparent, educational tools.

Special thanks to everyone who believes that **understanding how things work** is just as important as making them work.

<br>

---

<br>

<div align="center">

### 💖 Made with passion for the ML community

**If NumEdge helps you learn or build something awesome, please consider starring the repo!**

⭐ **Star** • 🍴 **Fork** • 📣 **Share**

---

[🐛 Report Bug](https://github.com/Nitin-Prata/numedge/issues) • [✨ Request Feature](https://github.com/Nitin-Prata/numedge/issues) • [💬 Discuss](https://github.com/Nitin-Prata/numedge/discussions)

---

**NumEdge** — *Machine Learning That Makes Sense*

</div>