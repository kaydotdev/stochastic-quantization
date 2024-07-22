from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="sq",
        fullname="Stochastic optimal quantization",
        description="Deep Embedded Clustering with Stochastic Optimal Quantization",
        long_description="""The challenge of working with high-dimensional data in unsupervised domains is well-known, 
particularly due to the 'curse of dimensionality.' As dimensionality increases, the data space volume expands rapidly, 
leading to sparse distributions and potential model overfitting. To address this issue, Deep Embedded Clustering (DEC) 
was proposed as a two-stage algorithm. The first stage employs a deep neural network's embedding model to learn low-
dimensional feature representations in metric space from high-dimensional data (e.g., images, text). The second stage 
applies clustering algorithms to analyze similarities between these low-dimensional features. However, this approach 
encounters difficulties in its second stage, where algorithms like K-means are used to solve non-convex optimization 
problems without robust theoretical convergence guarantees. This paper introduces a stochastic quantization model as 
an alternative second stage, demonstrating faster and more stable convergence on non-convex optimization problems. Our 
experimental evaluations on image and text data reveal significant improvements over existing one-stage models.""",
        version="1.0.0",
        author="Anton Kozyriev",
        author_email="a.kozyriev@kpi.ua",
        license="CC BY-NC-ND 4.0",
        url="https://github.com/kaydotdev/stochastic-quantization",
        platforms="Any",
        scripts=[],
        packages=find_packages(exclude=["tests", "notebooks"]),
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Education",
            "Natural Language :: English",
            "Natural Language :: Ukrainian",
            "Operating System :: OS Independent",
            "License :: Other/Proprietary License",
            "Topic :: Education",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        keywords=[
            "global stochastic optimization",
            "stochastic gradient descent",
            "deep embedded clustering",
            "non-convex optimization",
            "discrete quantization",
            "discrete clustering",
            "allocation problem",
            "deep learning",
            "similarity learning",
        ],
        install_requires=[
            "numpy>=1.26.4,<2",
            "scikit-learn=1.5.1,<2",
        ],
    )
