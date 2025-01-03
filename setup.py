"""Setup file for the rag package."""

from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.58.1",
        "sqlalchemy>=2.0.36",
        "psycopg2-binary>=2.9.10",
        "pgvector>=0.3.6",
        "python-dotenv>=1.0.1",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "pytest-cov===6.0.0"
        ]
    },
    python_requires=">=3.10",
)
