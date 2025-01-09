from setuptools import setup, find_packages

setup(
    name="vector-rag",
    version="0.1.0",
    description="RAG (Retrieval-Augmented Generation) System",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Richard Hightower",
    author_email="richardhightower@gmail.com",
    url="https://github.com/RichardHightower/rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.58.1,<2.0.0",
        "sqlalchemy>=2.0.36,<3.0.0",
        "psycopg2-binary>=2.9.10,<3.0.0",
        "pgvector>=0.3.6,<0.4.0",
        "python-dotenv>=1.0.1,<2.0.0",
    ],
    extras_require={
        "dev": [
            "black>=24.1.0,<25.0.0",
            "isort>=5.13.0,<6.0.0",
            "mypy>=1.8.0,<2.0.0",
            "pytest>=8.0.0,<9.0.0",
            "pytest-cov>=4.1.0,<5.0.0",
            "types-psycopg2>=2.9.21,<3.0.0"
        ]
    },
    python_requires=">=3.12,<3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
)