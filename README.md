# Robust Clustering with Stochastic Quantization

In this paper, we address the limitations of traditional vector quantization (clustering) algorithms, such as KMeans 
and its variants, which require the eager loading of all data points into memory, making them impractical for large 
datasets. Although variants like Mini-Batch KMeans partially mitigate this issue by reducing memory usage, they lack 
robust theoretical convergence guarantees due to the non-convex nature of clustering problems. To overcome these 
challenges, we introduce a stochastic quantization algorithm as a scalable alternative with strong theoretical 
convergence guarantees for solving unsupervised and semi-supervised learning problems. We demonstrate the 
effectiveness and robustness of this approach through an image classification task with partially labeled data. 
To address the problem of high dimensionality, we apply a deep embedded clustering technique to encode images 
into low-dimensional representations in a latent space, which we use to compare the efficiency of both the proposed 
and traditional quantization algorithms. Our experimental results reveal that the convergence speed of the introduced 
algorithm is on par with that of traditional algorithms.
