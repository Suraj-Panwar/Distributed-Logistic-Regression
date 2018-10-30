# Distributed-Logistic-Regression
Distributed Implementation of a Multiclass Logistic Regression,

This repository contains files for both local as well as Distributed Implementation of Logistic Regression.

1. Local Logistic Regression.py : 
   This file contains code for local implementation of Logistic Regression which trains model and returns test and train accuracy.

2. Tensorflow_Logistic_regression.py :
   This file contains code for Tensorflow based implementation of Logistic Regression using pickled parameter files from the local            execution for time saving due to similar code.

3. BSP_SGD.py : 
   This file contains code for execution of Distributed Synchronous execution of Logistic Regression, the number of replicas per model can    be changed by changing SyncReplicasOptimizer.

4. Async_SGD.py : 
   This file contains code for execution of Distributed Asynchronous execution of Logistic Regression.
4. SSP_SGD: 
   This file contains code for execution of Bounded Distributed Asynchronous execution of Logistic Regression.
