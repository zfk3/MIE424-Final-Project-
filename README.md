# MIE424-Final-Project
OPTIMIZING FOR FAIRNESS IN MACHINE LEARNING

#### Developed by Sajad Hashemi, Zahra Nadine Kandola, Scott Oxholm, and Rachael Walker

## Overview:
This repo is tangental to our paper for our MIE424 Final Project. In this project, we study the effects of adding fairness regularizers to the objective function when training binary classification models. Our goals are twofold. First, we seek to examine the fairness-accuracy tradeoffs of regularizing for a single definition of fairness. We then extend the literature by exploring the effectiveness  of regularizing for multiple definitions of fairness simultaneously. 

## Fairness Regularizers
''A Reductions Approach to Fair Classification''by Agerwal et al. implement fairness regularizers by using the true fairness definitions for equalized odds and demographic parity. They develop a novel formulation that can represent both of these definitions as a set of linear constraints. These constraints can then be synthesized into a regularization term in a model’s objective function. Which the approach taken by FairTorch, a repository that won first prize at the 2020 Global PyTorch Summer Hackathon in the Responsible AI category.  Their repository was intended to be used as a library of fairness regularizers that can be integrated into PyTorch models. The following notebooks were used to reimplement simplifications of FairTorch to be used in our project and can be found in the "Regularizers" folder. 

1. FairtorchDemo.ipynb: Demoing the FairTorch Library
2. equalized_odds.ipynb: Equalized Regularizer
3. Demographic Parity.ipynb: Demographic Parity Regularizer



The FairTorch implementation did not include any experimentation or exploration of real-world applications. We filled this gap by developing an experimentation framework, testing with real data, and developing intuition about how these regularizers impact model performance. 

## Exploratory Data Analysis 
We implemented our optimization problem using the famous COMPAS and Adult Income dataset. Our primliminary exploratory data anlysis, feature selection, and data processing can be found in the following notebook:

1. EDA.ipynb

## Experimentation Pipeline 
All experiments were run locally using CPU through an experimentation pipeline built on Google Colab in Python. The data processing and analysis portion of our pipeline was built with the standard pandas, plotly, and numpy packages. The ML models and testing in our pipeline were built using torch, sklearn, and fairlearn (a library to compute fairness metrics from ML model outputs). The experimentation pipeline consists of the following components: 

A. Data Loaders: To load and clean each dataset. 
B. Data Splitter: Splits the data into train/validation/test sets with fixed seeding. We used a single split for all of our testing (i.e. no cross-validation). This is because our experimentation involved training many different model variants and cross-validation for all of them would be impractical given the time. 
C. Models: Initialize the desired model architecture in Pytorch. 
D. Training and Testing Loop: framework to train, validate and test each model variant using a gradient descent approach. 
