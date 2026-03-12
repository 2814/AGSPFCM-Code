AGSP-FCM Clustering Algorithm (MATLAB Implementation)
Overview
This repository contains the MATLAB implementation of the Adaptive Gaussian Suppression based Possibilistic Fuzzy C-Means (AGSPFCM) clustering algorithm, SFCM, SPCM and SPFCM.
The proposed clustering framework enhances classical fuzzy clustering techniques by integrating:
Gaussian distance metric
Suppression mechanism
Possibilistic clustering concept
Noise and outlier resistance
The algorithm improves clustering robustness and accuracy compared with traditional methods such as FCM, PCM, and PFCM.
Algorithm Components

The proposed approach consists of two major stages:

1. Gaussian Distance Based FCM Initialization

The algorithm first applies a modified Fuzzy C-Means (FCM) clustering approach where the distance metric is replaced with a Gaussian based distance function.

This improves cluster separation and reduces sensitivity to noise.

Objective Function:

J = Σ Σ √(1 − A^(-b * d²))

where

A = Gaussian constant

b = distance scaling factor

d = Euclidean distance between data point and cluster center

2. Suppressed Possibilistic Clustering

After FCM initialization, the algorithm performs Suppressed PCM/FCM updates.

Key steps include:

Possibilistic membership calculation

Suppression mechanism to control noise points

Adaptive weighting of cluster memberships

Iterative update of cluster centers

Suppression parameter:

alpha = 0.88 (this could be tuned manually)

This prevents noisy data points from dominating cluster formation.

Datasets are uploaded

The current implementation uses the Glass dataset.

File used:

glassdata.txt

Structure:

Features  |  Class Label
x1 x2 x3 ... xn | C
Performance Metrics

The algorithm evaluates clustering performance using the following metrics:

Metric	Description
Misclassification	Number of incorrectly clustered samples
Accuracy	Clustering accuracy (%)
Silhouette Score	Cluster separation measure
Rand Index	Similarity between predicted and true clusters
Normalized Mutual Information (NMI)	Information similarity measure
SSE	Sum of squared clustering errors
MATLAB Requirements

Recommended MATLAB version:

MATLAB R2019 or later

Required helper functions:

clus_sse.m
adjrand.m
nmi1.m

These functions compute evaluation metrics.

How to Run the Code

Place all files in the same folder in matlab

Example structure:

AGSP-FCM
│
├── agspfcm.m
├── glassdata.txt
├── clus_sse.m
├── adjrand.m
├── nmi1.m

Open MATLAB.

Run the script:

agspfcm

The program will display:

Iteration objective values

Clustering accuracy

Misclassification count

Silhouette score

Rand Index

NMI

SSE

A graphical table of metrics will also be displayed.

Output Example
Accuracy = 84.67 %
Error Rate = 15.33 %

Total Misclassification = 12
Silhouette Score = 0.58
Rand Index = 0.73
NMI = 0.69
SSE = 143.22
Research Contribution

The proposed AGSPFCM algorithm provides the following contributions:

Gaussian based distance metric for improved cluster discrimination

Suppression mechanism for noise-robust clustering

Hybrid FCM-PCM optimization framework

Improved clustering accuracy for noisy datasets

Citation

If you use this implementation in your research, please cite the associated paper:

Author: Jyoti Arora et al.
Title: Adaptive Gaussian Suppression based Possibilistic Fuzzy C-Means Clustering
Journal: (To be added after publication)
