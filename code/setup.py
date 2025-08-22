from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="sqc",
        fullname="Stochastic Quasi-Gradient Clustering",
        description="A robust and scalable alternative to existing K-means solvers.",
        long_description="Stochastic Quasi-Gradient Clustering (alternatively termed Stochastic Quantization** within "
        "the machine learning domain), a robust and scalable alternative to existing K-means solvers. The algorithm is"
        " specifically designed to handle large-scale datasets while optimizing memory utilization during computational"
        " processes. It reframes the clustering problem as a stochastic transportation problem by minimizing the "
        "distance between elements of the original distribution {ξ} and atoms of the encoded discrete distribution {yₖ}.",
        version="1.1.0",
        author="Vladimir Norkin, Anton Kozyriev, Ostap Bodnar",
        author_email="a.kozyriev@kpi.ua",
        license="MIT",
        url="https://github.com/kaydotdev/stochastic-quantization",
        platforms="Any",
        scripts=[],
        python_requires=">=3.10",
        packages=find_packages(exclude=["tests", "tests.*"]),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Education",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: MIT License",
            "Topic :: Education",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        keywords=[
            "stochastic quantization",
            "clustering algorithms",
            "K-means",
            "stochastic gradient descent",
            "non-convex optimization",
        ],
        install_requires=[
            "numpy>=1.26.4,<2",
            "scikit-learn>=1.5.1,<2",
        ],
        extras_require={
            "faiss-cpu": ["faiss-cpu>=1.10.0,<2"],
            "faiss-gpu": ["faiss-gpu>=1.10.0,<2"],
            "progress": ["tqdm>=4.66.0,<5"],
            "all": ["sqc[faiss-cpu,faiss-gpu,progress]"],
        },
    )
