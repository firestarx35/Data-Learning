# Dimensionality reduction, Data assimilation and Domain decomposition.

This project focuses on dimensionality reduction, data assimilation, and domain decomposition techniques applied to wildfire data processing. The main objectives are to test the performance of PCA and autoencoders for data compression, compare their assimilation capabilities using Kalman Filters, and parallelize the task using domain decomposition.

### Installation
To run the notebooks and experiments in this repository, you need to have the following packages installed:
- Jupyter
- Scikit-learn
- PyTorch
- mpi4py

## Notebook Structure:
The repository is organized into several notebooks, each corresponding to a specific task. Here's a brief overview of the tasks and their respective notebooks:

### Dimensionality Reduction:

File: Dimensionality_reduction_PCA.ipynb
Objective: Evaluate the performance of Principal Component Analysis (PCA) for compressing wildfire data.

File: Dimensionality_reduction_Encoder.ipynb
Objective: Evaluate the performance of Convolutional Autoencoder for compressing wildfire data.

### Data Assimilation:

File: Assimilation_PCA.ipynb
Objective: Compare the performance of PCA in the context of assimilation using Kalman Filters.

File: Assimilation_Encoder.ipynb
Objective: Compare the performance of Convolutional AutoEncoder in the context of assimilation using Kalman Filters.

### Domain Decomposition:

File: Decomposition_mpi4py.ipynb
Objective: Implement domain decomposition using mpi4py to parallelize wildfire data processing.

### Note: Analysis for each task can be found at the end of respective notebooks.
