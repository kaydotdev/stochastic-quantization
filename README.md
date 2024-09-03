# Robust Clustering on High-Dimensional Data with Stochastic Quantization

[![Open paper in Colab](https://img.shields.io/badge/Colab-F9AB00?logo=googlecolab&color=525252)](https://colab.research.google.com/github/kaydotdev/stochastic-quantization/blob/master/code/notebooks/simlearning.ipynb)
[![Open paper in Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=Kaggle&logoColor=white)](https://www.kaggle.com/notebooks/welcome?src=https://github.com/kaydotdev/stochastic-quantization/blob/master/code/notebooks/simlearning.ipynb)

by Vladimir Norkin<sup>1,2</sup>, [Anton Kozyriev](mailto:a.kozyriev@kpi.ua)<sup>1</sup>

 - Igor Sikorsky Kyiv Polytechnic Institute, National Technical University of Ukraine, Kyiv, 03056, Ukraine
 - V.M.Glushkov Institute of Cybernetics, National Academy of Sciences of Ukraine, Kyiv, 03178, Ukraine

21 page, 5 figures, to be published in the [International Scientific Technical Journal "Problems of Control and 
Informatics"](https://jais.net.ua/)

## Categories

 - **Primary**: Machine Learning (cs.LG)
 - **Cross lists**: Optimization and Control (math.OC)
 - **MSC-class**: 90C15

## Abstract

This paper addresses the limitations of traditional vector quantization (clustering) algorithms, particularly K-Means 
and its variant K-Means++, and explores the Stochastic Quantization (SQ) algorithm as a scalable alternative for 
high-dimensional unsupervised and semi-supervised learning problems. Some traditional clustering algorithms suffer 
from inefficient memory utilization during computation, necessitating the loading of all data samples into memory, 
which becomes impractical for large-scale datasets. While variants such as Mini-Batch K-Means partially mitigate this 
issue by reducing memory usage, they lack robust theoretical convergence guarantees due to the non-convex nature of 
clustering problems. In contrast, the Stochastic Quantization algorithm provides strong theoretical convergence 
guarantees, making it a robust alternative for clustering tasks. We demonstrate the computational efficiency and rapid 
convergence of the algorithm on an image classification problem with partially labeled data, comparing model accuracy 
across various ratios of labeled to unlabeled data. To address the challenge of high dimensionality, we trained Triplet 
Network to encode images into low-dimensional representations in a latent space, which serve as a basis for comparing 
the efficiency of both the Stochastic Quantization algorithm and traditional quantization algorithms. Furthermore, we 
enhance the algorithm's convergence speed by introducing modifications with an adaptive learning rate.

## Keywords

stochastic quantization, clustering algorithms, stochastic gradient descent, 
non-convex optimization, deep metric learning, data compression

## License

This repository contains both software (source code) and an academic manuscript. Different licensing terms apply to 
these components as follows:
 1. Source Code: All source code contained in this repository, unless otherwise specified, is licensed under the MIT 
License. The full text of the MIT License can be found in the file [LICENSE.code.md](./code/LICENSE.code.md) in the `code` directory.
 2. Academic Manuscript: The academic manuscript, including all LaTeX source files and associated content (e.g. 
figures), is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License 
(CC BY-NC-ND 4.0). The full text of the CC BY-NC-ND 4.0 License can be found in the file 
[LICENSE.manuscript.md](./manuscript/LICENSE.manuscript.md) in the `manuscript` directory.
