from setuptools import setup, find_packages

setup(
    name="fairval",
    version="0.1.0",
    description="Fair Visual Active Learning with Information-Theoretic Guarantees",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "open-clip-torch>=2.20.0",
    ],
)
