# Udacity_Trafficsign_classifier
The project ues the convolutional network to classify the Germany trafficsign
The project contains the following steps:
* Load the data 
* Dataset Summary and Exploration
* Design and test a Model Architecture
* Test the Model on New Images


 ##Step 1 Load the data
 
 Use the `pickle` library, we import the data for this project. The code is as follow:
 
 ```
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

```

##Step 2 Dataset Summary and Exploration

throuth the code,we see the size of training data set,validation data set and testing data set.

we randomly picked 1 picture of different class and plot them.

we also plot the data distribution of both training data set and testing data set.

The code is as follow:
```
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation =len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape =X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print ('Test_data_shape=',X_test.shape)
```

The out put of the code:
![data_size](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic1.PNG)

```
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
plt.figure()
for class_idx in range(n_classes):
    cur_X = X_train[y_train ==class_idx]
    cur_img = cur_X[np.random.randint(len(cur_X))]
    ax = plt.subplot(4,12,class_idx+1)
    plt.imshow(cur_img)
    ax.set_title('{:02d}'.format(class_idx))
    plt.xticks(())
    plt.yticks(())


plt.figure()
train_distribution,test_distribution = np.zeros(n_classes),np.zeros(n_classes)    
for id in range(n_classes):
    train_distribution[id]=np.sum(y_train ==id)/n_train
    test_distribution[id] =np.sum(y_test ==id)/n_test
col_width = 0.5
bar_train = plt.bar(np.arange(n_classes),train_distribution,width = col_width,color = 'r')
bar_test =plt.bar(np.arange(n_classes)+col_width,test_distribution,width = col_width,color = 'b')
plt.ylabel('PERCENTAGE OF PRESENCE')
plt.xlabel('CLASS LABEL')
plt.title('Traffic Sign Distribution')
plt.xlim(0,45)
plt.ylim(0,0.06)
plt.legend((bar_train[0],bar_test[0]),('train set','test set'))
plt.show()
plt.show()

# Visualizations will be shown in the notebook.
%matplotlib inline
```

The output of the code is as follow:
![data_view](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic2.PNG)
![data_view](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic3.PNG)

##Step 3 Design and Test a Model Architecture

**Preprocess the Data set

To preprocess the data set, we mainly normalized the data, grayscale the data and then using shif and rotate to expand the number of the data, which could prevent overfitting while training the model.

```
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from scipy import ndimage
import cv2

###Normaliaze the data
def normalize_images(image_data):
    return image_data/255.
X_train = normalize_images(X_train)
X_valid = normalize_images(X_valid)
X_test = normalize_images(X_test)


###Change the pics to gray style 
def gray_scale(X):
    X = np.sum(X/3,axis =3, keepdims = True)
    return X

X_train = gray_scale(X_train)
X_valid = gray_scale(X_valid)
X_test = gray_scale(X_test)

##Change the order of the training data to prevent the model to learn the structure and order of the training data 
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

##expand the training data to 5 times scale by using shifting and rotating 
def expend_training_data(train_x, train_y):
    """
    Augment training data
    """
    expanded_images = np.zeros([train_x.shape[0] * 5, train_x.shape[1], train_x.shape[2]])
    expanded_labels = np.zeros([train_x.shape[0] * 5])

    counter = 0
    for x, y in zip(train_x, train_y):

        # register original data
        expanded_images[counter, :, :] = x
        expanded_labels[counter] = y
        counter = counter + 1

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = np.median(x)  # this is regarded as background's value

        for i in range(4):
            # rotate the image with random degree
            angle = np.random.randint(-15, 15, 1)
            new_img = ndimage.rotate(x, angle, reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

            # register new training data
            expanded_images[counter, :, :] = new_img_
            expanded_labels[counter] = y
            counter = counter + 1

    return expanded_images, expanded_labels

##Reshape the X_train data to fit the expand method
X_train = np.reshape(X_train,(-1, 32, 32))
X_train, y_train = expend_training_data(X_train, y_train)

##Change the X_train data shape back
X_train = np.reshape(X_train,[-1,32,32,1])

n_train = len(X_train)
n_classes = len(set(y_train))
n_test = len(X_test)
image_shape = X_train[0].shape


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```
The out put of this part is as bellow, you can see the data number of train set is expanding to 173995






