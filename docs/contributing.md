# Contributing to AILib

We welcome contributions to **AILib**! Whether you're fixing bugs, improving documentation, or adding new features, your help is invaluable. This guide outlines how you can contribute effectively to the AILib project.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Code of Conduct](#code-of-conduct)
- [Reporting Issues](#reporting-issues)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Style Guidelines](#style-guidelines)
- [Development Setup](#development-setup)
- [Testing](#testing)

## How to Contribute

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of the [AILib GitHub repository](https://github.com/Caua-ferraz/ailib) to create your own fork.

2. **Clone the Fork**

   ```bash
   git clone https://github.com/yourusername/ailib.git
   cd ailib
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Implement your feature, bug fix, or documentation improvement.

5. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**

   Navigate to your fork on GitHub and click the "Compare & pull request" button. Provide a clear description of your changes and submit the pull request.

## Code of Conduct

Please read and adhere to our [Code of Conduct](https://github.com/Caua-ferraz/ailib/blob/main/CODE_OF_CONDUCT.md) to ensure a welcoming and respectful environment for all contributors.

## Reporting Issues

If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/Caua-ferraz/ailib/issues) on the GitHub repository. Provide as much detail as possible to help us understand and address the problem effectively.

## Submitting Pull Requests

When submitting pull requests:

- **Ensure Tests Pass**: Make sure all existing tests pass. Add new tests for your changes if applicable.
- **Follow Style Guidelines**: Adhere to the project's coding style and guidelines.
- **Provide Descriptive Commits**: Use clear and descriptive commit messages.
- **Link Issues**: If your pull request addresses an existing issue, mention it in the description using `Closes #issue_number`.

## Style Guidelines

- **PEP8 Compliance**: Follow [PEP8](https://pep8.org/) style guidelines for Python code.
- **Docstrings**: Write clear and concise docstrings for all modules, classes, and functions.
- **Type Hinting**: Use type hints to improve code readability and maintainability.
- **Modular Code**: Keep code modular and reusable, avoiding unnecessary duplication.

## Development Setup

To set up your development environment:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ailib.git
   cd ailib
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Configure Logging and Error Handling**

   Set up your configuration file (e.g., `config.json`) as per the [Configuration Management](configuration.md) guidelines.

## Testing

**AILib** uses `pytest` for unit testing. To run tests:

```bash
pytest
```

Ensure that all tests pass before submitting your pull request. If you add new features, include corresponding tests to verify their functionality.

## Documentation

Improve and maintain the documentation to help users understand and utilize AILib effectively. Follow the [Sphinx](https://www.sphinx-doc.org/en/master/) guidelines if extending the documentation beyond markdown files.

## Getting Help

If you need assistance or have questions, feel free to [open an issue](https://github.com/Caua-ferraz/ailib/issues) or join our community discussions.

## Thank You!

Thank you for considering contributing to **AILib**! Your efforts help make the library more powerful and useful for everyone.
