# DSTIGCN_Master

DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for pedestrian trajectory prediction
Paper: [DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for pedestrian trajectory prediction](https://ieeexplore.ieee.org/document/10843981)))

The code and weights have been released, enjoy it！

## Introduction
Previous trajectory prediction methods face two common problems: 1. ignoring the joint modeling of pedestrians' complex spatial-temporal interactions, and 2. suffering from the long-tail effect, which prevents accurate capture of the diversity of pedestrians' future movements. 

To address these problems, we propose a Deformable Spatial-Temporal Interaction Graph Convolution Network (DSTIGCN). 
To solve problem 1, we design a deformable spatial-temporal interaction module. The module autonomously learns the spatial-temporal interaction relationships of pedestrians through the offset of multiple asymmetric deformable convolution kernels in both spatial and temporal dimensions, thereby achieving joint modeling of complex spatial-temporal interactions. 
To address problem 2, we introduce Latin hypercube sampling to sample the two-dimensional Gaussian distribution of future trajectories, thereby improving the multi-modal prediction effect of the model under limited samples.

## Method
The overall architecture of the model is shown in Fig. 1. DSTIGCN first constructs a spatial graph based on the position changes of pedestrians and then obtains the attention score matrix through attention mechanisms to preliminary represent the social interactions of pedestrians at each moment. To further achieve joint learning of pedestrian social and temporal interactions, we design a deformable spatial-temporal interaction module with multiple layers of deformable spatial-temporal interaction convolutions stacked together. The deformable spatial-temporal interaction convolution is optimized based on the deformable 3D convolution, which reduces the number of model parameters while avoiding excessive information redundancy. The deformable spatial-temporal interaction module processes the attention score matrix to capture the complex spatial-temporal interactions of pedestrians and then obtains the trajectory representation through graph convolution. Subsequently, we design a TAG-TCN to predict the parameters of the two-dimensional Gaussian distribution of future trajectories. Finally, we introduce Latin hypercube sampling to sample from the predicted two-dimensional Gaussian distribution, thereby achieving multi-modal prediction of pedestrian trajectories.
<img width="1008" alt="DSTIGCN" src="https://github.com/user-attachments/assets/e78300c4-f241-4892-8b8e-07dac2440ccd" />


Schematic diagram of deformable spatial-temporal interaction convolution.
<img width="1369" alt="可变形时空交互卷积示意图 - 修改5" src="https://github.com/user-attachments/assets/ce503b7a-fb94-4574-be54-23c3856f23c0" />



We also compare the trajectory visualization results of different sampling methods (MC, QMC, and LHS) when the number of samples is 20.
<img width="862" alt="DSTIGCN-预测分布对比 - 修改1" src="https://github.com/user-attachments/assets/586cbf19-997c-4f45-9c12-b6c5b3cee063" />


## Code Structure
checkpoint folder: contains the trained models

dataset folder: contains ETH and UCY datasets

model.py: the code of STIGCN

train.py: for training the code

test_Qmc.py: QMC sampling test code

test_Lhs.py: LHS sampling test code

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code


## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test_Lhs.py and test_Qmc.py


In addition, we apply QMC and LHS sampling methods to our public works STIGCN and IMGCN. We found that the LHS sampling method can also achieve better prediction results in these two works.

STIIGCN (Code: https://github.com/Chenwangxing/STIGCN_master;   Paper: [STIGCN: spatial–temporal interaction-aware graph convolution network for pedestrian trajectory prediction](https://link.springer.com/article/10.1007/s11227-023-05850-8))

The prediction errors of different sampling methods of STIGCN are shown in the following table:
| STIGCN  | ETH | HOTEL| UNIV| ZARA1 | ZARA2 | AVG |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| MC  | 0.58/0.96 | 0.30/0.44| 0.38/0.67| 0.28/0.47 | 0.23/0.42 | 0.35/0.59 |
| QMC  | 0.52/0.96 | 0.22/0.33| 0.31/0.56| 0.25/0.45 | 0.21/0.39 | 0.30/0.54 |
| LHS  | 0.43/0.68 | 0.24/0.48| 0.26/0.48| 0.22/0.41 | 0.17/0.32 | 0.26/0.47 |

IMGCN (Code: https://github.com/Chenwangxing/IMGCN_master;   Paper: [IMGCN: interpretable masked graph convolution network for pedestrian trajectory prediction](https://www.tandfonline.com/doi/abs/10.1080/21680566.2024.2389896))

The prediction errors of different sampling methods of IMGCN are shown in the following table:
| IMGCN  | ETH | HOTEL| UNIV| ZARA1 | ZARA2 | AVG |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| MC  | 0.61/0.82 | 0.31/0.45| 0.37/0.67| 0.29/0.51 | 0.24/0.42 | 0.36/0.57 |
| QMC  | 0.59/1.09 | 0.22/0.34| 0.31/0.58| 0.25/0.48 | 0.22/0.41 | 0.32/0.58 |
| LHS  | 0.54/1.03 | 0.23/0.45| 0.26/0.47| 0.21/0.39 | 0.18/0.34 | 0.28/0.54 |

## Acknowledgement
Some codes are borrowed from Social-STGCNN and SGCN. We gratefully acknowledge the authors for posting their code.

## Cite this article:

Chen W, Sang H, Wang J, et al. DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for Pedestrian Trajectory Prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2025.

