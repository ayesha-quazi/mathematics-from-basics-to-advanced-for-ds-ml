# Contributing to Mathematics for DS/ML

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Content Additions**
   - Add new mathematical concepts or topics
   - Create Jupyter notebooks with examples
   - Add Python implementations
   - Write detailed notes and explanations

2. **Improvements**
   - Fix errors in mathematical formulas or code
   - Improve explanations and clarity
   - Add visualizations
   - Enhance existing examples

3. **Documentation**
   - Improve README files
   - Add more detailed comments in code
   - Create tutorials or guides

4. **Bug Fixes**
   - Fix code errors
   - Correct mathematical mistakes
   - Fix broken links or references

## Getting Started

1. **Fork the Repository**
   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/mathematics-from-basics-to-advanced-for-ds-ml.git
   cd mathematics-from-basics-to-advanced-for-ds-ml
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-fix-name
   ```

## Content Guidelines

### Mathematical Content

- **Accuracy**: Ensure all mathematical formulas and concepts are correct
- **Clarity**: Explain concepts clearly, assuming the reader has basic knowledge
- **Examples**: Include worked examples for each concept
- **Citations**: Reference sources when appropriate

### Code Guidelines

- **Style**: Follow PEP 8 for Python code
- **Documentation**: Add docstrings to all functions and classes
- **Type Hints**: Use type hints where appropriate
- **Comments**: Add comments to explain complex logic
- **Testing**: Test your code to ensure it runs without errors

### Jupyter Notebooks

- **Structure**: Start with an introduction, followed by sections with clear headings
- **Explanations**: Alternate between markdown explanations and code cells
- **Visualizations**: Include plots and visualizations where helpful
- **Exercises**: Add practice exercises at the end
- **Clean Output**: Clear output cells before committing

### File Organization

Place your contributions in the appropriate directory:

```
01-Algebra-Basics/
02-Functions-and-Graphs/
03-Calculus/
    Differentiation/
    Integration/
    Optimization/
04-Linear-Algebra-for-ML/
05-Probability-and-Statistics/
06-Vector-Spaces-Eigenvalues-SVD/
07-Gradient-Descent-and-Optimization/
08-Numerical-Methods-and-Transformations/
notebooks/
```

## Pull Request Process

1. **Update Documentation**
   - Update README files if you add new content
   - Add your notebook to the notebooks/README.md list

2. **Test Your Changes**
   ```bash
   # Run Python scripts to ensure they work
   python your_script.py
   
   # Test notebooks (if applicable)
   jupyter nbconvert --execute --to notebook your_notebook.ipynb
   ```

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

4. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear title and description
   - Explain what changes you made and why

## Commit Message Guidelines

Use clear and descriptive commit messages:

- `Add: New content or feature` - for new additions
- `Fix: Issue description` - for bug fixes
- `Update: Content description` - for improvements
- `Docs: Documentation changes` - for documentation only changes

Examples:
```
Add: Fourier transform examples in numerical methods
Fix: Incorrect formula in gradient descent notes
Update: Improve PCA visualization in notebook
Docs: Add installation instructions to README
```

## Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone.

### Our Standards

Examples of behavior that contributes to a positive environment:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

Examples of unacceptable behavior:
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## Questions?

If you have questions or need help:
- Open an issue with the `question` label
- Reach out to maintainers

## Recognition

Contributors will be recognized in the project. Thank you for helping make this resource better for everyone learning mathematics for Data Science and Machine Learning!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
