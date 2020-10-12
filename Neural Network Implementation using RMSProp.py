# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:16:19 2020

@author: sshre
"""
#%%
#imports
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%
#this class handles pre-processing of the data
class dataManager:
    
    #initialize object
    def __init__(self, path, test_percentage):
        self.df = pd.read_csv(path)
        self.cleanData()
        self.scale()
        self.test_percentage = test_percentage
    
    #remove nulls and duplicates
    def cleanData(self):
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()

    #standardize the data
    def scale(self):
        scaler = StandardScaler()
        scaler.fit(self.df) 
        
        self.transformDF = pd.DataFrame(scaler.transform(self.df))
        colNames = self.df.columns
        self.transformDF.columns = colNames
        
        self.transformDF = self.transformDF.drop(columns=["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"])
        self.df = self.df[["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]]
        
        self.df = pd.concat([self.transformDF.reset_index(drop=True), self.df.reset_index(drop=True)], axis=1)
        
           
    #remove any unnecessary attributes
    def remove(self, cols):
        self.df = self.df.drop(self.df.columns[cols], axis=1)
    
    #function calls other functions which split X and Y, and also split into training and test data
    def prep(self):
        self.sepXY()
        self.trainTestSplit(self.test_percentage)        
    
    #seperate response from predictor attributes
    def sepXY(self):
        self.Y = self.df.iloc[:, [12]]
        self.X = self.df.iloc[:, 0:12]
    
    #training test split
    def trainTestSplit(self, test_percentage):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = test_percentage, random_state=1005)        
#%%
heartData = dataManager("https://sxs170830.s3.us-east-2.amazonaws.com/CS4372/heart_failure_clinical_records_dataset.csv", 0.25)
#the following code generates plots to look at the relationship between attributes, commented out because it takes a while to run
#correlation_matrix = heartData.df.iloc[:,0:12].corr().round(2)
#sns.heatmap(data=correlation_matrix, annot=True)

#none of the attributes are highly correlated, so we proceed
heartData.prep()
#%%
class NeuralNet:
    
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.Y_train = Y_train.copy()
        self.Y_test = Y_test.copy()
    
    def fit(self, activation = "sigmoid", h=4, max_iterations=1000, learning_rate=0.25, beta=0.9, epsilon=10**(-6)):
        
        self.input_layer_size = self.X_train.shape[1]
        if not isinstance(self.Y_train, np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.Y_train)

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((self.input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.ones((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        
        v_out = [[i] for i in np.zeros(h)]
        v_outb = [[i] for i in np.zeros(1)]
        
        v_hidden = np.zeros((self.X_train.T.shape[0], self.deltaHidden.shape[1]))
        v_hiddenb = [[i] for i in np.zeros(1)]
        
        self.h = h
        self.beta = beta
        self.trnErr = {}
        
        for iteration in range(max_iterations):
            #print(iteration)
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.Y_train), 2)
            self.trnErr[iteration] = np.sum(error)
            #TODO: I have coded the sigmoid activation, you have to do the rest
            self.backward_pass(out, activation)
            
            #rmsProp optimizer implemented below
            grad_out = np.dot(self.X_hidden.T, self.deltaOut)
            v_out_new = ([[beta*i[0]] for i in v_out]) + ( (1-beta) * (grad_out**2) )
            update_weight_output = (learning_rate/(np.sqrt(v_out_new + epsilon))) * grad_out
            
            grad_outb = np.dot(np.ones((np.size(self.X_train, 0), 1)).T, self.deltaOut)
            v_outb_new = ([[beta*i[0]] for i in v_outb]) + ( (1-beta) * (grad_outb**2) )
            update_weight_output_b = (learning_rate/(np.sqrt(v_outb_new + epsilon))) * grad_outb
            
            grad_hidden = np.dot(self.X_train.T, self.deltaHidden)
            v_hidden_new = ([[beta*i[0]] for i in v_hidden]) + ( (1-beta) * (grad_hidden**2) )
            update_weight_hidden = (learning_rate/(np.sqrt(v_hidden_new + epsilon))) * grad_hidden
            
            grad_hiddenb = np.dot(np.ones((np.size(self.X_train, 0), 1)).T, self.deltaHidden)
            v_hiddenb_new = ([[beta*i[0]] for i in v_hiddenb]) + ( (1-beta) * (grad_hiddenb**2) )
            update_weight_hidden_b = (learning_rate/(np.sqrt(v_hiddenb_new + epsilon))) * grad_hiddenb

            self.W_output += update_weight_output
            self.Wb_output += update_weight_output_b
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b
            
            v_out = v_out_new
            v_outb = v_outb_new
            v_hidden = v_hidden_new
            v_hiddenb = v_hiddenb_new
        
        self.activation = activation
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)[0]))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))
        
    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in_hidden = np.dot(self.X_train, self.W_hidden) + self.Wb_hidden
        self.X_hidden = self.__activation(in_hidden, activation)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        out = self.__activation(in_output, activation)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)
    
    def compute_output_delta(self, out, activation):
        delta_output = (self.Y_train - out) * (self.__activation_derivative(out, activation))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation):
        delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__activation_derivative(self.X_hidden, activation))

        self.deltaHidden = delta_hidden_layer
        
    #activation functions
    def __activation(self, x, activation):
        if activation == "sigmoid":
            return 1/(1 + np.exp(-x))
        elif activation == "tanh":
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        elif activation == "relu":
            new_x = x
            
            for i in np.arange(len(x)):
                for w in np.arange(len(x[i])):
                    new_x[i][w]=max(0, x[i][w])
                
            return new_x
        
    #activation functions derivatives
    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            return x*(1-x)
        elif activation == "tanh":
            return 1-(x**2)
        elif activation == "relu":
            new_x = x

            for i in np.arange(len(x)):
                for w in np.arange(len(x[i])):
                    if x[i][w]<0:
                        new_x[i][w]=0
                    else:
                        new_x[i][w]=1

            return new_x
    
    #plot training error vs. iteration
    def plotError(self):
        import matplotlib.pyplot as plt
        plt.plot(list(self.trnErr.keys()), list(self.trnErr.values()))#, 'b--')
        plt.ylabel('Training Loss')
        plt.xlabel('Number of Iterations')
        title = ('Activation:' + str(self.activation) +
                 '   Hidden Layers:' + str(self.h) + 
                 '   Iter:' + str(self.max_iterations) + 
                 '   Learning Rate:' + str(self.learning_rate))
        plt.title(title)
        plt.show()
    
    #convert activation function to prediction
    def __convertPreds(self, out):
        if self.activation == "sigmoid":
            return np.around(out, 0)
        elif self.activation == "tanh":
            return np.around(out, 0)
        elif self.activation == "relu":
            x = out
            
            for i in np.arange(len(out)):
                if out[i] == 0:
                    x[i] = 0
                else:
                    x[i] = 1
            
            return x
        
    def predict(self):
        #predict on training set
        in_hidden = np.dot(self.X_train, self.W_hidden) + self.Wb_hidden
        self.X_hidden = self.__activation(in_hidden, self.activation)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        trainOut = self.__activation(in_output, self.activation)
        
        self.trainError = np.sum(0.5 * np.power((trainOut - self.Y_train), 2))/len(trainOut)
        
        self.trainPreds = self.__convertPreds(trainOut)
        self.trainAccuracy = np.sum(self.Y_train.eq(self.trainPreds))/len(self.trainPreds)
        
        #predict on test set
        in_hidden = np.dot(self.X_test, self.W_hidden) + self.Wb_hidden
        self.X_hidden = self.__activation(in_hidden, self.activation)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        testOut = self.__activation(in_output, self.activation)
        
        self.testError = np.sum(0.5 * np.power((testOut - self.Y_test), 2))/len(testOut)
        
        self.testPreds = self.__convertPreds(testOut)
        self.testAccuracy = np.sum(self.Y_test.eq(self.testPreds))/len(self.testPreds)
        
    #print the metrics for the given model
    def getMetrics(self):
        self.predict()
        
        print("Neural Network Summary\n")
        print("Number of Iterations : ", self.max_iterations)
        print("Activation Function : ", self.activation)
        print("Learning Rate : ", self.learning_rate)
        print("Number of Hidden Layers : ", self.h, "\n")
        print("Scaled Training Error : ", self.trainError[0])
        print("Scaled Test Error : ", self.testError[0])
        print("Training Accuracy : ", self.trainAccuracy[0])
        print("Test Accuracy : ", self.testAccuracy[0])
#%%
#this object takes in different sets of parameters, fits to model and prints metrics for comparision purposes
class modelComparer:
    
    #initialize object
    def __init__(self, params, X_train, X_test, Y_train, Y_test):
        self.params = params
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
    
    #fit all models, store metrics in a dataframe
    def fitModels(self):
        
        #set up metric df
        self.metricdf = pd.DataFrame(columns = ['Activation', 'Hidden Layers', 'Iterations', 'Learning Rate', 
                                                'Scaled Training Error', 'Scaled Test Error', 
                                                'Training Accuracy', 'Test Accuracy'])
        
        #fit the model using each parameter set
        for paramSet in self.params:
            myNN = NeuralNet(self.X_train, self.X_test, self.Y_train, self.Y_test)
            myNN.fit(activation=paramSet[0], h=paramSet[1], max_iterations=paramSet[2], learning_rate=paramSet[3])
            myNN.plotError()
            myNN.predict()
            
            #append row to df
            i = len(self.metricdf) + 1
            self.metricdf.loc[i] = [paramSet[0], paramSet[1], paramSet[2], paramSet[3],
                                    myNN.trainError[0], myNN.testError[0], 
                                    myNN.trainAccuracy[0], myNN.testAccuracy[0]]
        
    #function prints metrics to console        
    def getResults(self):
        print(self.metricdf)
    #function writes metrics to excel file
    def writeResults(self, path):
        self.metricdf.to_excel(path)
#%%
params = [['sigmoid', 4, 500, 0.1], 
          ['sigmoid', 4, 500, 0.25],
          ['sigmoid', 6, 500, 0.25],
          ['sigmoid', 4, 1000, 0.25],
          ['tanh', 4, 500, 0.1], #best
          ['tanh', 4, 500, 0.25],
          ['tanh', 6, 500, 0.1],
          ['tanh', 6, 500, 0.25],
          ['relu', 4, 250, 0.25], 
          ['relu', 4, 250, 0.1],
          ['relu', 4, 500, 0.1],
          ['relu', 6, 500, 0.1]]

#model comparer code is commented out because it takes time to run, and requires excel output
#selector = modelComparer(params, heartData.X_train, heartData.X_test, heartData.Y_train, heartData.Y_test)
#selector.fitModels()
#selector.getResults()
#%%
#NEED TO SPECIFY PATH
#selector.writeResults("C:/Users/sshre/Downloads/modelComparision.xlsx")
#%%
#fit the best model and look at it in detail
myNetwork = NeuralNet(heartData.X_train, heartData.X_test, heartData.Y_train, heartData.Y_test)
myNetwork.fit(activation="tanh", h = 4, max_iterations = 500, learning_rate = 0.1)
myNetwork.plotError()
myNetwork.predict()
myNetwork.getMetrics()