# Installation

**AILib** can be installed using `pip`, the Python package installer. Follow these steps to set up AILib in your environment.

## Requirements

- **Python 3.7** or later
- **pip** (Python package installer)

## Installing AILib

To install the latest stable version of AILib, run the following command:

```bash
pip install ailib
```

For the latest development version, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/yourusername/ailib.git
```

## Verifying the Installation

After installation, you can verify that AILib is correctly installed by running:

```python
import ailib
print(ailib.__version__)
```

This should print the version number of AILib.

## Optional Dependencies

Some features of AILib require additional dependencies. To install AILib with all optional dependencies, use:

```bash
pip install ailib[all]
```

## Installing for Development

If you plan to contribute to AILib or need the latest unreleased version, you can install it in editable mode:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ailib.git
   cd ailib
   ```

2. Install in editable mode:

   ```bash
   pip install -e .
   ```

## Troubleshooting

If you encounter any issues during installation, please check our [Troubleshooting Guide](troubleshooting.md) or open an issue on our [GitHub repository](https://github.com/Caua-ferraz/ailib/issues).
