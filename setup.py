import os
from setuptools import setup, find_packages

# Use absolute path for requirements.txt
requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')

if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        required = f.read().splitlines()
else:
    print(f"Warning: {requirements_path} not found. Proceeding without dependencies.")
    required = []

# Remove scikit-learn from the requirements list
required = [req for req in required if not req.startswith('scikit-learn')]

setup(
    name="ailib",
    version="0.3.0",
    packages=find_packages(),
    install_requires=required,
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive AI library including traditional ML models and LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ailib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)