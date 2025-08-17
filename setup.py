from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-new-journey",
    version="1.0.0",
    author="AI新生教程",
    author_email="contact@ai-new-journey.com",
    description="《AI新生：从零到一的智能革命之旅》完整代码实现",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AI-New-Journey",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="ai, machine learning, deep learning, tutorial, 人工智能, 机器学习, 深度学习, 教程",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/AI-New-Journey/issues",
        "Source": "https://github.com/yourusername/AI-New-Journey",
        "Documentation": "https://github.com/yourusername/AI-New-Journey/wiki",
    },
)
