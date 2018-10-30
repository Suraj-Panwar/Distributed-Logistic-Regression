# Distributed-Logistic-Regression
Distributed Implementation of a Multiclass Logistic Regression,

This repository contains files for both local as well as Distributed Implementation of Logistic Regression.

1. Local Logistic Regression.py : 
   This file contains code for local implementation of Logistic Regression which trains model and returns test and train accuracy.
   
2. BSP_SGD.py : 
   This file contains code for execution of Distributed Synchronous execution of Logistic Regression, the number of replicas per model can    be changed by changing SyncReplicasOptimizer.
