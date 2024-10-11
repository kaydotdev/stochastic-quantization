# Robust Clustering on High-Dimensional Data with Stochastic Quantization

[![Arxiv](https://img.shields.io/badge/arXiv-2409.02066-B21A1B)](https://doi.org/10.48550/arXiv.2409.02066)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?logo=googlecolab&color=525252)](https://colab.research.google.com/github/kaydotdev/stochastic-quantization/blob/master/code/notebooks/simlearning.ipynb)
[![Open In Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=Kaggle&logoColor=white)](https://www.kaggle.com/notebooks/welcome?src=https://github.com/kaydotdev/stochastic-quantization/blob/master/code/notebooks/simlearning.ipynb)

by [Anton Kozyriev](mailto:a.kozyriev@kpi.ua)<sup>1</sup>, [Vladimir Norkin](mailto:v.norkin@kpi.ua)<sup>1,2</sup>

 - Igor Sikorsky Kyiv Polytechnic Institute, National Technical University of Ukraine, Kyiv, 03056, Ukraine
 - V.M.Glushkov Institute of Cybernetics, National Academy of Sciences of Ukraine, Kyiv, 03178, Ukraine

22 pages, 5 figures, to be published in the [International Scientific Technical Journal "Problems of Control and Informatics"](https://jais.net.ua/)

## Introduction

This paper addresses the limitations of conventional vector quantization algorithms, particularly K-Means and its 
variant K-Means++, and investigates the Stochastic Quantization (SQ) algorithm as a scalable alternative for 
high-dimensional unsupervised and semi-supervised learning tasks. Traditional clustering algorithms often suffer 
from inefficient memory utilization during computation, necessitating the loading of all data samples into memory, 
which becomes impractical for large-scale datasets. While variants such as Mini-Batch K-Means partially mitigate this 
issue by reducing memory usage, they lack robust theoretical convergence guarantees due to the non-convex nature of 
clustering problems. In contrast, the Stochastic Quantization algorithm provides strong theoretical convergence 
guarantees, making it a robust alternative for clustering tasks. We demonstrate the computational efficiency and rapid 
convergence of the algorithm on an image classification problem with partially labeled data, comparing model accuracy 
across various ratios of labeled to unlabeled data. To address the challenge of high dimensionality, we employ a 
Triplet Network to encode images into low-dimensional representations in a latent space, which serve as a basis for 
comparing the efficiency of both the Stochastic Quantization algorithm and traditional quantization algorithms. 
Furthermore, we enhance the algorithm's convergence speed by introducing modifications with an adaptive learning rate.

## License

This repository contains both software (source code) and an academic manuscript. Different licensing terms apply to 
these components as follows:
 1. Source Code: All source code contained in this repository, unless otherwise specified, is licensed under the MIT 
License. The full text of the MIT License can be found in the file [LICENSE.code.md](./code/LICENSE.code.md) in the
`code` directory.
 2. Academic Manuscript: The academic manuscript, including all LaTeX source files and associated content (e.g. 
figures), is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License 
(CC BY-NC-ND 4.0). The full text of the CC BY-NC-ND 4.0 License can be found in the file 
[LICENSE.manuscript.md](./manuscript/LICENSE.manuscript.md) in the `manuscript` directory.

## Citation

```bib
@misc{Kozyriev_Norkin_2024,
    title={Robust Clustering on High-Dimensional Data with Stochastic Quantization}, 
    author={Anton Kozyriev and Vladimir Norkin},
    year={2024},
    eprint={2409.02066},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2409.02066},
}
```
