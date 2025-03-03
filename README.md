# Stochastic Vector Quantization

This repository explores an implementation of the **Stochastic K-means algorithm** (also referred to as the 
**Stochastic Quantization algorithm**), a robust and scalable alternative to existing K-means solvers, designed to 
handle large datasets and utilize memory more efficiently during computation. The implementation examines the 
application of the algorithm to high-dimensional unsupervised and semi-supervised learning tasks. The repository 
contains both a Python package for reproducing experimental results and a LaTeX manuscript documenting the theoretical 
and experimental outcomes of the Stochastic Quantization algorithm. The Python package continues to evolve 
independently of the research documentation; therefore, to reproduce specific results presented in the paper, 
researchers should refer to the commit hash mentioned in the description.

## Research Articles

### Robust Clustering on High-Dimensional Data with Stochastic Quantization

[![DOI](https://img.shields.io/badge/DOI-10.34229/1028--0979--2025--1--3-blue.svg)](https://doi.org/10.34229/1028-0979-2025-1-3)
[![Arxiv](https://img.shields.io/badge/arXiv-2409.02066-B21A1B)](https://doi.org/10.48550/arXiv.2409.02066)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?logo=googlecolab&color=525252)](https://colab.research.google.com/github/kaydotdev/stochastic-quantization/blob/master/code/notebooks/quantization.ipynb)
[![Open In Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=Kaggle&logoColor=white)](https://www.kaggle.com/notebooks/welcome?src=https://github.com/kaydotdev/stochastic-quantization/blob/master/code/notebooks/quantization.ipynb)

by [Anton Kozyriev](mailto:a.kozyriev@kpi.ua)<sup>1</sup>, [Vladimir Norkin](mailto:v.norkin@kpi.ua)<sup>1,2</sup>

1. Igor Sikorsky Kyiv Polytechnic Institute, National Technical University of Ukraine, Kyiv, 03056, Ukraine
2. V.M.Glushkov Institute of Cybernetics, National Academy of Sciences of Ukraine, Kyiv, 03178, Ukraine

Published in the International Scientific Technical Journal 
["Problems of Control and Informatics"](https://jais.net.ua/). This paper addresses the inherent limitations of 
traditional vector quantization (clustering) algorithms, particularly K-means and its variant K-means++, and 
investigates the stochastic quantization (SQ) algorithm as a scalable alternative methodology for high-dimensional 
unsupervised and semi-supervised learning problems.

**Latest commit hash**

[ed22ae0b5507564d917b57d4cbdea952cc134d77](https://github.com/kaydotdev/stochastic-quantization/commit/ed22ae0b5507564d917b57d4cbdea952cc134d77)

**Citation**

```bib
@article{Kozyriev_Norkin_2025,
	title        = {Robust clustering on high-dimensional data with stochastic quantization},
	author       = {Kozyriev, Anton and Norkin, Vladimir},
	year         = {2025},
	month        = {Feb.},
	journal      = {International Scientific Technical Journal "Problems of Control and Informatics"},
	volume       = {70},
	number       = {1},
	pages        = {32–48},
	doi          = {10.34229/1028-0979-2025-1-3},
	url          = {https://jais.net.ua/index.php/files/article/view/438},
	abstractnote = {&amp;lt;p&amp;gt;This paper addresses the limitations of traditional vector quantization (clustering) algorithms, particularly K-means and its variant K-means++, and explores the stochastic quantization (SQ) algorithm as a scalable alternative for high-dimensional unsupervised and semi-supervised learning problems. Some traditional clustering algorithms suffer from inefficient memory utilization during computation, necessitating the loading of all data samples into memory, which becomes impractical for large-scale datasets. While variants such as mini-batch K-means partially mitigate this issue by reducing memory usage, they lack robust theoretical convergence guarantees due to the non-convex nature of clustering problems. In contrast, SQ-algorithm provides strong theoretical convergence guarantees, making it a robust alternative for clustering tasks. We demonstrate the computational efficiency and rapid convergence of the algorithm on an image classification problem with partially labeled data. To address the challenge of high dimensionality, we trained Triplet Network to encode images into low-dimensional representations in a latent space, which serve as a basis for comparing the efficiency of both SQ-algorithm and traditional quantization algorithm.&amp;lt;/p&amp;gt;},
}
```

## Getting Started

Before working with the source code, it is important to note that the Python package in the repository is intended 
**SOLELY FOR EXPERIMENTAL PURPOSES** and is not production-ready. To proceed with this project, follow the instructions 
below to configure your environment, install the necessary dependencies, and execute the code to reproduce the results 
presented in the paper.

### Dependencies

The installation process requires a Conda package manager for managing third-party dependencies and virtual 
environments. A step-by-step guide on installing the CLI tool is available on the official 
[website](https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links). The third-party dependencies used 
are listed in the [environment.yml](./environment.yml) file, with the corresponding licenses in the 
[NOTICES](./NOTICES) file.

### Installation

Clone the repository (alternatively, you can download the source code as a 
[zip archive](https://github.com/kaydotdev/stochastic-quantization/archive/refs/heads/master.zip)):

```shell
git clone https://github.com/kaydotdev/stochastic-quantization.git
cd stochastic-quantization
```

then, create a Conda virtual environment and activate it:

```shell
conda env create -f environment.yml
conda activate stochastic-quantization
```

### Reproducing the Results

Use the following command to install the core `sq` package with third-party dependencies, run the test suite, compile 
LaTeX files, and generate results:

```shell
make all
```

Produced figures and other artifacts (except compiled LaTeX files) will be stored in the [results](./results) 
directory. Optionally, use the following command to perform the actions above without LaTeX file compilation:

```shell
make -C code all
```

To automatically remove all generated results and compiled LaTeX files produced by scripts, use the following command:

```shell
make clean
```

## License

This repository contains both software (source code) and an academic manuscript. Different licensing terms apply to 
these components as follows:

1. Source Code: All source code contained in this repository, unless otherwise specified, is licensed under the MIT 
License. The full text of the MIT License can be found in the file [LICENSE.code.md](./code/LICENSE.code.md) in the 
`code` directory.

2. Academic Manuscript: The academic manuscript, including all LaTeX source files and associated content (e.g., 
figures), is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License 
(CC BY-NC-ND 4.0). The full text of the CC BY-NC-ND 4.0 License can be found in the file 
[LICENSE.manuscript.md](./manuscript/LICENSE.manuscript.md) in the `manuscript` directory.