# Multi Layered Random Forests

For linked phenomena with similar input and output spaces and limited individual data, we seek to develop a process to link datasets through multilayered analysis. Intuitively, borrowing from ideas in the pre-traing literature, a simple way to ensure cross-dimensional regularization is to require cross-dimensional information leakage when we determine the structure of the ensemble trees. For example, we may require that ensemble trees be characterized by a common rootstock  of depth $\delta$ across all or some dimensions. After the common rootstock, dimension-specific scions, or deeper branches, may be used to adapt the regression function to coordinate-specific heterogeneity.

Library contains a modified version of Breiman and Cutler's standard random forest (the base library was pulled from an MIT python implementation), in which a designated source indicator feature is restricted from being used in splits until a depth or node size condition is met.

TMB_Experiments were initial attempts at understanding how this novel hyperparameter affects regression learning. However, TMB data was noisy and forests were consistently underfitting.

Switching to classification, we look at artificially generated datasets, as well as real heart disease data.
