# DSTIGCN_Master

The code will be made public immediately after searching the article "DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for pedestrian trajectory prediction"!

See you later!

## Introduction
Previous trajectory prediction methods face two common problems: 1. ignoring the joint modeling of pedestrians' complex spatial-temporal interactions, and 2. suffering from the long-tail effect, which prevents accurate capture of the diversity of pedestrians' future movements. 

To address these problems, we propose a Deformable Spatial-Temporal Interaction Graph Convolution Network (DSTIGCN). 

To solve problem 1, we design a deformable spatial-temporal interaction module. The module autonomously learns the spatial-temporal interaction relationships of pedestrians through the offset of multiple asymmetric deformable convolution kernels in both spatial and temporal dimensions, thereby achieving joint modeling of complex spatial-temporal interactions. 

To address problem 2, we introduce Latin hypercube sampling to sample the two-dimensional Gaussian distribution of future trajectories, thereby improving the multi-modal prediction effect of the model under limited samples.

## Method
The overall architecture of the model is shown in Fig. 1. DSTIGCN first constructs a spatial graph based on the position changes of pedestrians and then obtains the attention score matrix through attention mechanisms to preliminary represent the social interactions of pedestrians at each moment. To further achieve joint learning of pedestrian social and temporal interactions, we design a deformable spatial-temporal interaction module with multiple layers of deformable spatial-temporal interaction convolutions stacked together. The deformable spatial-temporal interaction convolution is optimized based on the deformable 3D convolution, which reduces the number of model parameters while avoiding excessive information redundancy. The deformable spatial-temporal interaction module processes the attention score matrix to capture the complex spatial-temporal interactions of pedestrians and then obtains the trajectory representation through graph convolution. Subsequently, we design a TAG-TCN to predict the parameters of the two-dimensional Gaussian distribution of future trajectories. Finally, we introduce Latin hypercube sampling to sample from the predicted two-dimensional Gaussian distribution, thereby achieving multi-modal prediction of pedestrian trajectories.
<img width="1008" alt="DSTIGCN" src="https://github.com/user-attachments/assets/e78300c4-f241-4892-8b8e-07dac2440ccd" />



We compare our method with Social-STGCNN, SGCN, and RDGCN through trajectory visualization, as shown in Fig. 2.
![DSTIGCN-单条轨迹 - 修改1](https://github.com/user-attachments/assets/1d7bcb0b-9670-46af-8ca9-ace215a04e82)

We also compare the trajectory visualization results of different sampling methods (MC, QMC, and LHS) when the number of samples is 20, as shown in Fig. 3.
<img width="862" alt="DSTIGCN-预测分布对比 - 修改1" src="https://github.com/user-attachments/assets/586cbf19-997c-4f45-9c12-b6c5b3cee063" />


## Code Structure
checkpoint folder: contains the trained models

dataset folder: contains ETH and UCY datasets

model.py: the code of STIGCN

test_Qmc.py: QMC sampling test code

test_Lhs.py: LHS sampling test code

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code


\begin{table}[!t]
\caption{Ablation experiments with different sampling methods. The lower, the better. Bold: best.}
\centering
\label{table8}
\setlength{\tabcolsep}{1mm}{
\begin{tabular}{ccccccc}
\hline
          & ETH       & HOTEL     & UNIV      & ZARA1     & ZARA2     & AVG       \\ \hline
MC        & 0.60/1.00 & 0.33/0.56 & 0.37/0.70 & 0.28/0.50 & 0.24/0.43 & 0.36/0.64 \\
QMC       & 0.53/0.99 & \textbf{0.22}/\textbf{0.34} & 0.31/0.57 & 0.24/0.45 & 0.21/0.41 & 0.30/0.55 \\
Our (LHS) & \textbf{0.43}/\textbf{0.70} & \textbf{0.22}/0.41 & \textbf{0.25}/\textbf{0.45} & \textbf{0.20}/\textbf{0.37} & \textbf{0.17}/\textbf{0.32} & \textbf{0.25}/\textbf{0.45} \\ \hline
\end{tabular}}
\end{table}


## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test_Lhs.py and test_Qmc.py



## Acknowledgement
Some codes are borrowed from Social-STGCNN and SGCN. We gratefully acknowledge the authors for posting their code.


