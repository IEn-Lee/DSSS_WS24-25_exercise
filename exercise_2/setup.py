from setuptools import setup, find_packages

setup(
    name="math_quiz",
    version="0.1",
    description="A simple math quiz game with a unit test suite",
    author="I-EN LEE",
    author_email="ian10301004@gmail.com",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "math_quiz=math_quiz:math_quiz",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
