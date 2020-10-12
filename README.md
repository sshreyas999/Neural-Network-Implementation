# Neural Network Implementation

A simple neural network implementation with one hidden layer. Activation functions like sigmiod, tanh, and ReLu are supported, and the RMSProp optimizer is used to carry out backpropagation. 

## Quick Links
[Code](https://github.com/sshreyas999/Neural-Network-Implementation/blob/main/Neural%20Network%20Implementation%20using%20RMSProp.py)
[Data](https://github.com/sshreyas999/Neural-Network-Implementation/blob/main/heart_failure_clinical_records_dataset.csv)
[Output File - Logs, Proofs & Analysis](https://github.com/sshreyas999/Neural-Network-Implementation/blob/main/Neural%20Network%20Implementation%20-%20Metrics%2C%20Plots%20%26%20Proofs.pdf). 

## Prerequisites

The implemenation was written using the Spyder IDE, and basic packages like pandas, numpy, seaborn, and matplotlib are required. However, to write out metrics to an excel sheet we also need to install the xlrd and openpyxl package.

## Code Breakdown

Complete code can be found [here](https://github.com/sshreyas999/Neural-Network-Implementation/blob/main/Neural%20Network%20Implementation%20using%20RMSProp.py).

Breakdown of classes:

1. The **dataManager** class is responsible for reading in the raw data and preparing the data for learning. It cleans the dataset, seperates attributes and splits the set into test and train.

2. The **NeuralNet** class is responsible for learning using a given activation function (sigmoid, tanh, or ReLu), and an added optimization of RMSProp. It performs all the computations for learning and also calculates and stores metrics.

3. The **modelComparer** class takes in different sets of parameters and trains the model repeatedly. It stores all the metrics so that we can compare and take a look at which model is the best. The code for this has also been commented out since it takes a while to run. The results are included in the log file.

## Dataset

The dataset is hosted on S3, but can also be found [here](https://github.com/sshreyas999/Neural-Network-Implementation/blob/main/heart_failure_clinical_records_dataset.csv). 

The main goal is to predict the proabability of heart failure, encoded as **DEATH_EVENT**. There are 299 observations with 13 attributes per observation. Kindly note that this is only one example that is used to demonstrate the implementation. Other datasets can be loaded via the dataManager class.

## Analysis & Derivations

Apart from the **modelComparer** class in the code, a separate analysis has been carried out with the Heart Failure dataset. Multiple trials have been conducted and documented in the output file. A thorough explanation of the optimizer is also provided in the output file which can be found [here](https://github.com/sshreyas999/Neural-Network-Implementation/blob/main/Neural%20Network%20Implementation%20-%20Metrics%2C%20Plots%20%26%20Proofs.pdf).
