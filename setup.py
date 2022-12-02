from setuptools import setup, find_packages

setup(
    name="survey-enhance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "microdf_python",
        "torch",
        "policyengine_core",
    ],
    extras_require={
        "dev": [
            "black",
            "jupyter-book",
        ],
    },
)
