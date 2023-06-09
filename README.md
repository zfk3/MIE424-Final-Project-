# MIE424-Final-Project
OPTIMIZING FOR FAIRNESS IN MACHINE LEARNING

#### Developed by Sajad Hashemi, Zahra Nadine Kandola, Scott Oxholm, and Rachael Walker

## Overview:
This repository supports [our paper](https://github.com/zfk3/MIE424-Final-Project-/blob/main/MIE424%20Final%20Report.pdf) for our MIE424 Final Project. In this project, we study the effects of adding fairness regularizers to the objective function when training binary classification models. Our goals are twofold. First, we seek to examine the fairness-accuracy tradeoffs of regularizing for a single definition of fairness. We then extend the literature by exploring the effectiveness  of regularizing for multiple definitions of fairness simultaneously. 

## Running the Project:
To run our project and replicate results, you can run either (1) `DataPipeline_COMPAS.ipynb` or (2) `DataPipeline_ADULT.ipynb` in Google Colab. Further details pretaining to each notebook are explained below. 

The datasets we used can be collected from the following sources. 

Datasets: 
1. [COMPAS](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv)
2. [Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)

For ease of implementation, we have included our cleaned version of the adult dataset. The COMPAS dataset can be extracted directly from the link in the code.

## Fairness Regularizers:
''A Reductions Approach to Fair Classification'' by Agerwal et al. implement fairness regularizers by using the true fairness definitions for equalized odds and demographic parity. They develop a novel formulation that can represent both of these definitions as a set of linear constraints. These constraints can then be synthesized into a regularization term in a model’s objective function. This is the approach taken by FairTorch, a repository that won first prize at the 2020 Global PyTorch Summer Hackathon in the Responsible AI category.  Their repository was intended to be used as a library of fairness regularizers that can be integrated into PyTorch models. The following notebooks were used to reimplement simplifications of FairTorch to be used in our project and can be found in the "Regularizers" folder. 

1. `FairtorchDemo.ipynb`: Demoing the FairTorch Library
2. `equalized_odds.ipynb`: Equalized Odds Regularizer
3. `Demographic Parity.ipynb`: Demographic Parity Regularizer

The FairTorch implementation did not include any experimentation or exploration of real-world applications. We filled this gap by developing an experimentation framework, testing with real data, and developing intuition about how these regularizers impact model performance. 

## Exploratory Data Analysis: 
We implemented our optimization problem using the famous COMPAS and Adult Income datasets. Our primliminary exploratory data anlysis, feature selection, and data processing can be found in the following notebook:
 
1. `EDA.ipynb`

## Experimentation Pipeline: 
All experiments were run locally using CPU through an experimentation pipeline built on Google Colab in Python. The data processing and analysis portion of our pipeline was built with the standard pandas, plotly, and numpy packages. The ML models and testing in our pipeline were built using torch, sklearn, and fairlearn (a library to compute fairness metrics from ML model outputs). The experimentation pipeline consists of the following components: 

A. Data Loaders: To load and clean each dataset. 


B. Data Splitter: Splits the data into train/validation/test sets with fixed seeding. We used a single split for all of our testing (i.e. no cross-validation). This is because our experimentation involved training many different model variants and cross-validation for all of them would be impractical given the time. 


C. Models: Initialize the desired model architecture in Pytorch that has been integrated with the definitions of fairness. 


D. Training and Testing Loop: framework to train, validate and test each model variant using a gradient descent approach. 

This can be found in the following notebook: 
1. `DataPipeline_COMPAS.ipynb`: Priamry notebook used for training the COMPAS data. 
2. `DataPipeline_ADULT.ipynb`: Primary notebook used for training the Adult Income data. 

Note: both pipelines contain the same code that was seperated to allow for parallel training and to store the distinct hyperparameters for each dataset. *The `DataPipeline_COMPAS.ipynb` file has additional commenting for understanding*. 

Previous iterations of these notebooks can also be found in the "Past Data Pipelines" folders. Note, these may not be fully commented as they are draft iterations. 

## Results and Analysis: 
In our analysis we aimed to understand use of 1) a single fairness regularizer, 2) combining fairness regularizers, specifically looking at the fairness-accuracy tradeoff. 

Our main notebooks for this portion of the project are as follows: 
1. `Analysis_1_Single_Regularizer.ipynb` which contains all graphs from Section 5.3.1 of our report. 
2. `Analysis_2_Combining_Regularizers.ipynb` which contains all graphs from Section 5.3.2 of the report. 

These can be run using the `results_compas.csv` and `results_adult.csv` data files. 

Additional notebooks with our rough work can be found in the following notebooks in the "Past Results Notebooks" folder. 
1. `Results_Graphs.ipynb`: Model performance and fairness metrics plotted against various alpha values for single regularizers to understand their impact on these values. 
2. `Results_Combined.ipynb`: Heatmaps for various tradeoff parameter values and differnet fairness and model performance metrics to understand the combined effect of fairness regularizers. 
3. `Alpha_Tuning_COMPAS.ipynb`: Accuracy vs. Fairness graphs to understand the trade off for different alpha values for the COMPAS dataset. Heatmaps for percent difference for accuracy, demographic parity, and equalized odds from the baseline model for various alpha values. 
4. `Alpha_Tuning_Adult.ipynb`: Accuracy vs. Fairness graphs to understand the trade off for different alpha values for the Adult dataset. Heatmaps for percent difference for accuracy, demographic parity, and equalized odds from the baseline model for various alpha values. 

