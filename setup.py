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
        "ipython",
        "scikit-learn",
        "tensorboard",
    ],
    extras_require={
        "dev": [
            "black",
            "jupyter-book",
            "sphinx>=4.5.0,<5",
            "sphinx-argparse>=0.3.2,<1",
            "sphinx-math-dollar>=1.2.1,<2",
            "furo<2023",
        ],
    },
)
