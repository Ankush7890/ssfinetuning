from setuptools import setup, find_packages

setup(
    name="ssfinetuning",
    version="0.1.0", 
    author="Ankush Checkervarty",
    author_email="ankushc48@gmail.com",
    description="A package for fine tuning of pretrained NLP transformers using Semi Supervised Learning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP transformers huggingface deep learning pytorch",
    #url="https://github.com/huggingface/transformers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.6.0",
    install_requires=['torch>=1.7','transformers==4.2.2', 'datasets==1.5'],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    test_suite='tests'
)