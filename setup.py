"""EthicAgent - Setup Configuration."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ethicagent",
    version="1.0.0",
    author="EthicAgent Research Team",
    author_email="ethicagent@research.org",
    description=(
        "A Context-Aware Neuro-Symbolic Framework for Ethical "
        "Autonomous Decision-Making in Agentic AI Systems"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/radhahai/agent",
    project_urls={
        "Homepage": "https://github.com/radhahai/agent",
        "Repository": "https://github.com/radhahai/agent",
        "Issues": "https://github.com/radhahai/agent/issues",
        "Documentation": "https://github.com/radhahai/agent/tree/main/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ethicagent=ethicagent.orchestrator:main",
        ],
    },
)
