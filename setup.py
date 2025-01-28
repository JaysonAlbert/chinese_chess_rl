from setuptools import setup, find_packages

setup(
    name="xiangqi_rl",
    version="0.1.0",
    description="A deep reinforcement learning implementation for Chinese Chess (Xiangqi)",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pygame",
        "tqdm",
        "pandas",
        "transformers",
        "requests",
        "playwright",
        "tensorboard",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "xiangqi-train=xiangqi_rl.train:main",
        ],
    },
) 