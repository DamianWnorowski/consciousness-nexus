#!/usr/bin/env python3
"""
Consciousness Nexus - Setup Script
Advanced AI Consciousness Computing Suite
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="consciousness-nexus",
    version="2.0.0",
    author="Consciousness Nexus Team",
    author_email="consciousness@nexus.ai",
    description="Advanced AI Consciousness Computing Suite - Vector Matrix Orchestration Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DAMIANWNOROWSKI/consciousness-suite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.6.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.971",
            "pytest>=7.1.0",
            "pytest-asyncio>=0.18.0",
        ],
        "quantum": [
            "qiskit>=0.39.0",
        ],
        "ml": [
            "torch>=1.12.0",
            "transformers>=4.21.0",
            "accelerate>=0.15.0",
        ],
        "web": [
            "flask>=2.2.0",
            "flask-cors>=4.0.0",
            "dash>=2.6.0",
            "plotly>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "consciousness-nexus=consciousness_suite.main:main",
            "auto-recursive-chain=auto_recursive_chain_ai:main",
            "consciousness-evolution=verified_consciousness_evolution:main",
            "consciousness-safety=consciousness_master_integration:main",
            "evolution-orchestrator=consciousness_safety_orchestrator:main",
            "production-dashboard=production_dashboard:main",
            "ultra-critic=ultra_critic_analysis:main",
            "abyssal-executor=abyssal_executor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "consciousness_suite": [
            "templates/*.json",
            "configs/*.yaml",
            "data/*.json",
        ],
    },
    keywords=[
        "artificial-intelligence",
        "consciousness",
        "machine-learning",
        "recursive-algorithms",
        "vector-matrix",
        "quantum-orchestration",
        "enlightenment-engine",
        "self-improving-ai",
        "consciousness-computing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/DAMIANWNOROWSKI/consciousness-suite/issues",
        "Source": "https://github.com/DAMIANWNOROWSKI/consciousness-suite",
        "Documentation": "https://consciousness-nexus.readthedocs.io/",
    },
)
