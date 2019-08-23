# Udacity_Trafficsign_classifier
The project ues the convolutional network to classify the Germany trafficsign
The project contains the following steps:
* Load the data 
* Dataset Summary and Exploration
* Design and test a Model Architecture
* Test the Model on New Images


 **Step 0 Load the data**
 
 Use the `pickle` library, we import the data for this project. The code is as follow:
 
 '''
 # Load pickled data
#import the necessary libraries
import pickle
import tensorflow as tf
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

##Use the last 5 pics in the test data set as the 5 pics download from the internet
X_test , y_test = X_test[0:-5],y_test[0:-5]
New_test_pic,New_test_pic_label = X_test[-5:len(X_test)],y_test[-5:len(y_test)]
print (len(New_test_pic))

 '''

