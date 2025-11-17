# 🤝 Contributing to NumEdge

First off, **thank you** for considering contributing to NumEdge! 🎉

NumEdge is built by the community, for the community. Whether you're fixing a typo, reporting a bug, or implementing a new algorithm, every contribution makes this project better.

---

## 🌟 Ways to Contribute

### 🐛 Reporting Bugs

Found a bug? Help us squash it!

**Before reporting:**
- Check if the issue already exists in [Issues](https://github.com/Nitin-Prata/numedge/issues)
- Make sure you're using the latest version

**When reporting, include:**
- NumEdge version
- Python version
- Operating system
- Clear steps to reproduce
- Expected vs actual behavior
- Code snippet (if applicable)
- Error messages/stack traces

[**→ Report a Bug**](https://github.com/Nitin-Prata/numedge/issues/new?labels=bug)

---

### 💡 Suggesting Features

Have an idea? We'd love to hear it!

**Before suggesting:**
- Check if it's already suggested in [Discussions](https://github.com/Nitin-Prata/numedge/discussions)
- Consider if it aligns with NumEdge's philosophy (transparency, education, simplicity)

**When suggesting, explain:**
- The problem you're trying to solve
- Your proposed solution
- Alternative solutions you've considered
- Why this benefits NumEdge users

[**→ Suggest a Feature**](https://github.com/Nitin-Prata/numedge/discussions/new?category=ideas)

---

### 📖 Improving Documentation

Documentation is just as important as code!

**Ways to help:**
- Fix typos or unclear explanations
- Add code examples
- Improve docstrings
- Create tutorials or guides
- Translate documentation

All documentation improvements are welcome, no matter how small.

---

### ✨ Contributing Code

Ready to write some code? Awesome!

#### 🛠️ Development Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/numedge.git
cd numedge

# 3. Add upstream remote
git remote add upstream https://github.com/Nitin-Prata/numedge.git

# 4. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install in development mode with dev dependencies
pip install -e ".[dev]"

# 6. Install pre-commit hooks (optional but recommended)
pre-commit install
```

#### 🔄 Development Workflow

```bash
# 1. Create a new branch
git checkout -b feature/my-awesome-feature

# 2. Make your changes
# ... edit files ...

# 3. Run tests
pytest tests/

# 4. Check code style
black src/numedge tests/
flake8 src/numedge tests/

# 5. Commit your changes
git add .
git commit -m "Add awesome feature"

# 6. Push to your fork
git push origin feature/my-awesome-feature

# 7. Open a Pull Request on GitHub
```

---

## 📋 Code Guidelines

### 🎨 Code Style

NumEdge follows [PEP 8](https://pep8.org/) with these tools:

- **Black** — Code formatting
- **Flake8** — Linting
- **isort** — Import sorting

```bash
# Format code
black src/numedge tests/

# Check linting
flake8 src/numedge tests/

# Sort imports
isort src/numedge tests/
```

### 🏗️ Code Principles

1. **Readability First**
   - Clear variable names
   - Comprehensive docstrings
   - Comments for complex logic

2. **Pure NumPy/Python**
   - No Cython, C extensions, or compiled code
   - Educational transparency is the goal

3. **Type Hints**
   - Use type hints for function signatures
   - Helps with IDE support and documentation

4. **Error Handling**
   - Validate inputs early
   - Raise informative exceptions
   - Use NumEdge's custom exception classes

### 📝 Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1: int, param2: str) -> float:
    """
    Brief description of what this function does.
    
    Longer description if needed. Explain the algorithm,
    mathematical background, or important considerations.
    
    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.
    
    Returns
    -------
    float
        Description of return value.
    
    Raises
    ------
    ValueError
        If param1 is negative.
    
    Examples
    --------
    >>> result = my_function(5, "hello")
    >>> print(result)
    42.0
    
    Notes
    -----
    Any additional mathematical details, references,
    or implementation notes.
    
    References
    ----------
    .. [1] Author Name, "Paper Title", Journal, Year
    """
    pass
```

---

## 🧪 Testing

### Writing Tests

- Every new feature needs tests
- Tests should cover normal cases, edge cases, and error cases
- Use descriptive test names

```python
# tests/test_my_feature.py
import pytest
import numpy as np
from numedge.models import MyNewModel


def test_my_model_basic_fit():
    """Test that model can fit simple data."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    
    model = MyNewModel()
    model.fit(X, y)
    
    assert hasattr(model, 'is_fitted_')


def test_my_model_predict_shape():
    """Test that predictions have correct shape."""
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    X_test = np.random.randn(20, 5)
    
    model = MyNewModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    assert predictions.shape == (20,)


def test_my_model_invalid_input():
    """Test that model raises error on invalid input."""
    model = MyNewModel()
    
    with pytest.raises(ValueError):
        model.fit([[1, 2]], [1, 2, 3])  # Mismatched shapes
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_my_feature.py

# Run with coverage
pytest --cov=numedge tests/

# Run with verbose output
pytest -v
```

---

## 🎯 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated (if needed)
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated (if applicable)

### PR Checklist

1. **Title**: Clear, descriptive title
   - ✅ "Add Ridge regression with L2 regularization"
   - ❌ "Fixed stuff"

2. **Description**: Explain what and why
   ```markdown
   ## Changes
   - Added Ridge regression implementation
   - Includes alpha parameter for regularization strength
   
   ## Motivation
   Ridge regression is essential for handling multicollinearity
   in linear models.
   
   ## Testing
   - Added unit tests in tests/test_ridge.py
   - Verified against scikit-learn results
   ```

3. **Link Issues**: Reference related issues
   ```markdown
   Closes #42
   Related to #15
   ```

4. **Keep it Focused**: One feature/fix per PR
   - Easier to review
   - Easier to merge
   - Easier to revert if needed

---

## 🚀 Adding New Algorithms

Want to implement a new ML algorithm? Great! Here's the workflow:

### 1. Choose Your Algorithm

- Check if it's already in the roadmap
- Ensure it fits NumEdge's philosophy
- Discuss in [Discussions](https://github.com/Nitin-Prata/numedge/discussions) first

### 2. Implement Base Structure

```python
# src/numedge/models/your_category/your_algorithm.py

import numpy as np
from numedge.core.base import BaseEstimator
from numedge.core.mixins import RegressorMixin  # or ClassifierMixin


class YourAlgorithm(BaseEstimator, RegressorMixin):
    """
    Brief description of your algorithm.
    
    Parameters
    ----------
    param1 : type
        Description.
    
    Attributes
    ----------
    fitted_param_ : type
        Description of fitted attribute.
    
    Examples
    --------
    >>> model = YourAlgorithm(param1=value)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, param1=default_value):
        self.param1 = param1
    
    def fit(self, X, y):
        """Fit the model."""
        # Validate inputs
        X, y = self._validate_data(X, y)
        
        # Your algorithm implementation
        # ...
        
        # Mark as fitted
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        # Check if fitted
        self._check_is_fitted()
        
        # Validate input
        X = self._validate_data(X)
        
        # Make predictions
        # ...
        
        return predictions
    
    def get_search_space(self):
        """Return recommended hyperparameter search space."""
        return {
            'param1': [value1, value2, value3],
            # ...
        }
```

### 3. Write Comprehensive Tests

```python
# tests/test_your_algorithm.py

import pytest
import numpy as np
from numedge.models.your_category import YourAlgorithm


class TestYourAlgorithm:
    
    def test_basic_fit(self):
        """Test basic fitting."""
        pass
    
    def test_predict(self):
        """Test predictions."""
        pass
    
    def test_score(self):
        """Test scoring."""
        pass
    
    def test_invalid_input(self):
        """Test error handling."""
        pass
    
    def test_search_space(self):
        """Test hyperparameter search space."""
        pass
```

### 4. Add Documentation

- Update README.md algorithms list
- Add example in `examples/` directory
- Write docstrings with mathematical details

### 5. Submit PR

Follow the PR process above!

---

## 🎓 Learning Resources

Want to contribute but not sure where to start?

### Good First Issues

Look for issues labeled [`good first issue`](https://github.com/Nitin-Prata/numedge/labels/good%20first%20issue) — these are beginner-friendly!

### Learn the Codebase

1. Start with `src/numedge/core/base.py` — understand BaseEstimator
2. Read simple algorithms like `LinearRegression`
3. Move to more complex ones like `RandomForest`

### Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [How to Write Good Commit Messages](https://chris.beams.io/posts/git-commit/)

---

## 💬 Communication

### Where to Ask Questions

- **General questions**: [Discussions](https://github.com/Nitin-Prata/numedge/discussions)
- **Bug reports**: [Issues](https://github.com/Nitin-Prata/numedge/issues)
- **Direct contact**: nitinpratap997@gmail.com

### Code of Conduct

We're committed to providing a welcoming and inclusive environment.

**Our Standards:**
- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

**Unacceptable:**
- Harassment, discrimination, or offensive comments
- Trolling or insulting remarks
- Personal or political attacks

Violations may result in temporary or permanent ban.

---

## 🏆 Recognition

All contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation

Significant contributions may earn you:
- Maintainer status
- Your name in the README
- Special mentions in blog posts/announcements

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Every contribution, no matter how small, makes NumEdge better.

Thank you for being part of this journey! 💖

---

<div align="center">

**Questions?** Open a [Discussion](https://github.com/Nitin-Prata/numedge/discussions) or email nitinpratap997@gmail.com

[Back to README](README.md) • [View Issues](https://github.com/Nitin-Prata/numedge/issues) • [Start Discussion](https://github.com/Nitin-Prata/numedge/discussions)

</div>