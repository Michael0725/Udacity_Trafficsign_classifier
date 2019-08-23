# Udacity_Trafficsign_classifier
The project ues the convolutional network to classify the Germany trafficsign
The project contains the following steps:
* Load the data 
* Dataset Summary and Exploration
* Design and test a Model Architecture
* Test the Model on New Images


 ## Step 1 Load the data
 
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

## Step 2 Dataset Summary and Exploration

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

## Step 3 Design and Test a Model Architecture

**Preprocess the Data set**

To preprocess the data set, we mainly normalized the data, grayscale the data and then using shif and rotate to expand the number of the data, which could prevent overfitting while training the model.

The first time I tried to train the model, I find that the accuracy is very low, and after 5 or 6 epochs, the validation accuracy start to decrease.It obvious that overfitting happed. So try to find ways to prevent the overfitting.
During the comparison, I find that, using the data augmentation makes the huge difference.
I also tried to change the data to u-0 by using (X_data-128)/128, But it seems the result is not so good.
I checked from the internet and find someguy use the method of dividing 255. I tried the method, and the result seems good.
Since it takes a long time to train a model, so I did not combine all the potential method.
As for gray the image, I just want to decrease the weights to prevent overfitting.
From the final accuracy, it seems good to combine the 3 methods together.


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

![data_view](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic4.PNG)

**Model architecture**

The model architecture is as follow:

we defined the epochs to 10 and the batch size to 128

The neural net work architecture is as follow:

**Input:**

    32x32x1
    
**Layer1:**

    Convolutional matrix: 5x5x1x64    →   28x28x64
    Relu to activate the nods
    Max pooling : 1x2x2x1          →   14x14x64
    
**Layer2:** 

    Convolutional matrix: 5x5x64x128   →   10x10x128
    Relu to activate the nods
    Max pooling : 1x2x2x1           →   5x5x128
    
**Layer3:**

    Flatten the layer:　5x5x128  → 3200
    
**layer4:** 

    Full connected layer 3200   →512
    Relu to activate the nods
    Dropout layer
    
**layer5:**

    Full connected layer 512  →  120
    Relu to activate the nods
    Dropout layer
    
**Last layer:**

    Full connected layer 120  →  43
    return the result of logits
  
  The code is as bellow:
  ```
  ### Define your architecture here.
### Feel free to use as many code cells as needed.
EPOCHS = 10
BATCH_SIZE = 128
from tensorflow.contrib.layers import flatten
def mynet(image_data):
    mu = 0
    sigma = 0.1
    keep_prob = 0.5
    
    #  Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x64.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 64), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(64))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #  Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x64. Output = 14x14x64.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 128), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(128))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x128. Output = 5x5x128.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x128. Output = 3200.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 3200. Output = 512.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(3200, 512), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(512))
 fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    
    #Drop out
    fc1    =tf.nn.dropout(fc1,keep_prob)

    # Layer 4: Fully Connected. Input = 512. Output = 120.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(512, 120), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(120))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2,keep_prob)

    # Layer 5: Fully Connected. Input = 120   Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(120, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
    
  ```
  
  **Train Validata and test the model**
  
  Using the traing data to train the defined neural network The training accuracy arrived to about `99.7%` The validation accuracy arrived to about `97.5% `The testing accuracy arrived to `96.2%`
  
  The epochs are `10`, the batch size is `128` and the optimizer is `AdamOptimizer`
  
  The code is bellow:
  ```
  ### features and labels here
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

### Train your model here.
rate = 0.001
logits = mynet(x)
prediction = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

### Calculate and report the accuracy on the training and validation set.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        training_accuracy = evaluate(X_train, y_train)   
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    test_accuracy = evaluate(X_test,y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
        
    saver.save(sess, './lenet')
    print("Model saved")
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
  ```
  
  You can see the testing processing and the training accuracy, validation accuracy and testing accuracy as bellow picture
  ![data_view](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic5.PNG)
  

**The process of increasing the test accracy**

Actually, at first, the accuracy on both training data and validation data is very low. The overfitting happened.
So I tried to find ways to prevent the overfitting.
At first, I tried to use small converlutional filters, which are 5x5X6 , but it seems the result is not so good.
So I think I need to increase the data numbers, I tried to shift and rotate the images to increase the scale of training data, the accuracy increased.
I continue to use the grayscale to change the pic to graystyle and I also add the dropout layers.it seems the result become better and better.
Finally I got a good training result.
## Test the Model on New Images

**Load the new images**

We use the last 5 pictures of the testing data_set as the 5 new pics.

we put out the class name of the 43 traffic signs and also put out the labels and class name of the 5 new pictures 

The code is as follow:

```
### Load the images and plot them here.
for i in range(len(New_test_pic)):
    plt.subplot(1,5,i+1)
    plt.imshow(New_test_pic[i])
    plt.xticks(())
    plt.yticks(())
    
    
import pandas as pd
csv_file = pd.read_csv('signnames.csv')
class_name = csv_file['SignName'] 


class_name_of_last_5pics = []
for i in range(len(New_test_pic)):
    class_name_of_last_5pics.append(class_name[New_test_pic_label[i]]) 
print ("The class name  =",class_name)
print ("The label of last 5 pics of the testing data =",New_test_pic_label)
print ('The class name of last 5 pics of the testing data = ',class_name_of_last_5pics)
### Feel free to use as many code cells as needed.

```

We can see all the class types of the 43 traffic sign and the 5 new image's label and class name

![data_view](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic6.PNG)

**Predict the Sign Type for Each Image and output the top 5 possible class predicted by the network**

we predict the Sign type for the 5 new images and output the top 5 possible class.

the code is as follow:
```
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
## Calculate the top 5 possible class predicted by the neuralnetwork
New_test_pic_norm = normalize_images(New_test_pic)
New_test_pic_gray = gray_scale(New_test_pic_norm)
with tf.Session() as sess:
    saver.restore(sess, './lenet') 
    predict = sess.run(prediction,feed_dict={x:New_test_pic_gray})
    top5 = sess.run(tf.nn.top_k(tf.constant(predict), k=5))
print (top5)
plt.figure
for i in range(len(New_test_pic_gray)):
    print ('The top 5 possible class predicted by the neuralnetwork of the',i+1,'th picture is\n')
    print (class_name[top5[1][i]])  
    plt.subplot(5,2,2*i+1)
    plt.imshow(New_test_pic[i])
    plt.xticks(())
    plt.yticks(())
    plt.subplot(5,2,2*i+2)
    plt.bar(csv_file['ClassId'],predict[i])


```

The out put is as follow:

![data_view](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic8.PNG)
![data_view](https://github.com/Michael0725/Udacity_Trafficsign_classifier/blob/master/pictures/pic9.PNG)


**Output the accuracy of the prediction**

The code is as follow:
```
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
##Calculate the accuracy of the prediction
Correct_item = 0
for i in range(len(top5[1])):
    if New_test_pic_label[i] ==top5[1][i][0]:
        Correct_item += 1

Accuracy = Correct_item*20
print ("Accuracy = ",Accuracy,'%')
```

And you can see the accuracy is 100%















