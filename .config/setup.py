from setuptools import setup, find_packages

setup(
    name="42",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers==2.2.2",
        "qdrant-client==1.7.0",
        "tree-sitter==0.20.3",
        "unstructured==0.11.1",
        "hdbscan==0.8.33",
        "umap-learn==0.5.6",
        "typer==0.12.3",
        "fastapi==0.110.1",
        "uvicorn==0.27.1",
        "pydantic==2.6.1",
        "rich==13.7.0",
        "loguru==0.7.2",
        "redis==5.0.1",
        "httpx==0.27.0",
        "python-dotenv==1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest==8.0.0",
            "pytest-asyncio==0.23.5",
            "black==24.1.1",
            "ruff==0.2.1",
            "mypy==1.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "42=42.cli:main",
        ],
    },
    python_requires=">=3.9",
    author="42 Team",
    description="Autonomous, self-learning AI system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 