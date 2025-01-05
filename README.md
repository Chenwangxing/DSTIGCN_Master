# DSTIGCN_Master

The code will be made public immediately after searching the article "DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for pedestrian trajectory prediction"!

See you later!

## Introduction
Previous trajectory prediction methods face two common problems: 1. ignoring the joint modeling of pedestrians' complex spatial-temporal interactions, and 2. suffering from the long-tail effect, which prevents accurate capture of the diversity of pedestrians' future movements. 

To address these problems, we propose a Deformable Spatial-Temporal Interaction Graph Convolution Network (DSTIGCN). To solve problem 1, we design a deformable spatial-temporal interaction module. The module autonomously learns the spatial-temporal interaction relationships of pedestrians through the offset of multiple asymmetric deformable convolution kernels in both spatial and temporal dimensions, thereby achieving joint modeling of complex spatial-temporal interactions. To address problem 2, we introduce Latin hypercube sampling to sample the two-dimensional Gaussian distribution of future trajectories, thereby improving the multi-modal prediction effect of the model under limited samples.

## Method
The overall architecture of the model is shown in Fig. 1. DSTIGCN first constructs a spatial graph based on the position changes of pedestrians and then obtains the attention score matrix through attention mechanisms to preliminary represent the social interactions of pedestrians at each moment. To further achieve joint learning of pedestrian social and temporal interactions, we design a deformable spatial-temporal interaction module with multiple layers of deformable spatial-temporal interaction convolutions stacked together. The deformable spatial-temporal interaction convolution is optimized based on the deformable 3D convolution, which reduces the number of model parameters while avoiding excessive information redundancy. The deformable spatial-temporal interaction module processes the attention score matrix to capture the complex spatial-temporal interactions of pedestrians and then obtains the trajectory representation through graph convolution. Subsequently, we design a TAG-TCN to predict the parameters of the two-dimensional Gaussian distribution of future trajectories. Finally, we introduce Latin hypercube sampling to sample from the predicted two-dimensional Gaussian distribution, thereby achieving multi-modal prediction of pedestrian trajectories.

[Figure 1.pdf](https://github.com/user-attachments/files/18310111/Figure.3.-.2.pdf)


## Code Structure
dataset folder: contains ETH and UCY datasets




## Acknowledgement
Some codes are borrowed from Social-STGCNN and SGCN. We gratefully acknowledge the authors for posting their code.


