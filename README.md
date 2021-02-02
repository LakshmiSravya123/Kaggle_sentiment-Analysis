# Kaggle_sentiment-Analysis


## Description
The goal is to design a machine learning algorithm using classifiers to predict the pre-defined set of topics (classes). The data set given consists of 20 different classes and around 3500 messages as train data. The test data has 1500 Reddit messages whose classes are needed to be predicted with maximum accuracy. I have tried implementing around 6 classifiers which include Naïve Bayes, SVM, NN, Logistic, Random Forest, and voting classifier along with SVM

## Installation
Download the kaggle.py

## Dataset
data_train.pkl
data_test.pkl

 
## Import
import numpy as np
import pickle

## Classifiers

1) Multinomial Naïve Bayes - Naïve classification method is based on the Bayes rule. The advantage of Naïve Bayes is it works well on extremely large features and it rarely over fits the data. It is very fast in training and prediction for the amount of data it can handle.

2) Random forest - It is a classification algorithm consisting of many decision trees. Each tree in a random forest spits out a class prediction and the class with the most votes becomes model predictions.

3) Voting Classifier using SVM - A voting classifier model combines multiple models into a single model which is stronger than individual models. SVM algorithm determines the best decision boundary between vectors that belong to a given class and vectors that do not belong to it. The Ensemble voting classifier uses the arg max function which predicts the classes sharply than SVM.

4)  Logistic Regression - logistic regression is a discriminative classifier that solves the task by learning, from a training set, a vector of weights, and a bias term. It measures the relationship between the dependent and independent variables using logistic/sigmoid functions by measuring the probabilities.

## Test_train Split

The training set is split into 80% of training data and 20% as validation data set with which the learning of the classifiers took place and then the whole test data is passed to the classifier for predictions.

## Results
Classifiers | Hyper parameters tuned | Time taken | Accuracy
--- | --- | ---| --- |
Naïve Bayes|Alpha=0.2305|1 min|58.6|
SVM |C=1, t=0.01, gamma=auto|30.7 mins|57.3|
Sequential NN |batch size=32,epochs=15,verbose=1|Each epoch-773s,(199.2 mins)|53.37|
Logistic|verbose=1, solver=’saga’,C=5,max_iter=3000|8.2 mins| 56.7|
Random Forest Classifier|n_estimators=2000,min_samples_split=5min_samples_leaf=1,max_features=’sqrt’,max_depth=80,bootstrap=True|15.5 mins|48.6|
Voting Classifier along with SVM|Voting=hard|5 hrs|59.7

In the above table, it is understood the voting classifier using SVM has the highest accuracy and the random forest for the considered hyperparameters has the minimum accuracy

## Ideas for Improvement
1) The training set should be a larger one.
2) More usage of ensemble methods such as bagging, boosting and staking
3) More classifiers should be used in the voting classifier
4) N-grams has to be used in pre-processing instead of one gram/bigram


## Language used
Python

