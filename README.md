# Robust Clustering on High-Dimensional Data with Stochastic Quantization


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
