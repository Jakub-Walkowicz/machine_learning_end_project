# Bank Marketing Campaign Prediction Model

## Project Overview
This project focuses on predicting the success of bank telemarketing campaigns using machine learning models. The goal is to forecast whether a client will subscribe to a term deposit based on various features including client demographics, contact details, and economic indicators.

## Key Features
- Analysis of numerical and categorical variables' impact on campaign success
- Implementation of three machine learning models: KNN, SVM, and Bagging Classifier 
- SHAP value analysis for model interpretability
- Data balancing techniques using RandomUnderSampler
- Feature importance evaluation

## Data Preprocessing
- Removal of 'duration' variable (not known before making a call)
- One-hot encoding of categorical variables
- Data standardization using RobustScaler
- Creation of binary features from historical contact data
- Training data balancing while preserving test set distribution

## Models
- KKNN (Weighted K-Nearest Neighbors)
- SVM (Support Vector Machine)  
- Bagging Classifier with Decision Tree base model

## Key Findings
- Seasonal factors (May and August) strongly influence campaign success
- Contact method and history are significant predictors
- Client characteristics have moderate impact on success probability
- Multiple contact attempts don't significantly increase success probability

## Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy  
- shap
- matplotlib

## Author
Jakub Walkowicz

## Dataset Citation
[Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.

Available at:
- [PDF](http://hdl.handle.net/1822/14838)
- [BIB](http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt)
