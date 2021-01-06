# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/epochs.png "Model training exploration"
[image4]: ./examples/probabilities.png "Traffic Signs probabilities"
[image5]: ./examples/visualizingNN.png "Visualizing Neural Network"
[image6]: ./examples/data.png "Data conclusion"
[image7]: ./examples/architecture.png "Model Architecture"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rcgonzsv/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing samples per class in the data.

![alt text][image1]

As we can observe, the data is imbalanced. It will be great to balance the dataset so that the network can avoid skewed learning towards classes that have more data. There are 2 common ways you can mitigate this problem:

**Undersampling**, ramdomly resample some of the observations from the majority class in order to match the numbers with the menority class.

**Oversampling**, generating data augmentation such as random scaling, cropping, rotation, etc.

In this case and despite the data for the model is small, we are not generating data augmentation since model is training pretty well, and validation suggest that not significantly marginal benefit will comoe out from this. However, for most robust models, it would be required.


![alt text][image6]

### Design and Test a Model Architecture

![alt text][image7]
*Input and output shapes for each layer can be found in detail in the notebook and architecture description below.*

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to pursue a simple pre-procesing for the data, reshaping, normalizing and converting the images to grayscale in order to prepare the inputs to my chosen architecture.

Normalization is choosen over standardization since normalization is good to use when you know that the distribution of your data does not follow a Gaussian distribution (as shown above) and standardization, on the other hand, can be helpful in cases where the data follows a Gaussian distribution and calculate z-scores centering values around the mean and standard deviation.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 2x2    	| 1x1 stride, valid padding, outputs 28x28x1  	|
| RELU - Max Pooling	|												|
| Convolution 2x2    	| 1x1 stride, valid padding, outputs 14x14x1  	|
| RELU - Max Pooling	|												|
| Layer 3  flattening   | Dropout = 0.8 in training, outuput: 600x300  	|
| Layer 4 Fully Conn.   | ReLu activation,           outuput: 300x100  	|
| Layer 5 Fully Conn.   | output:100 x n_classes = 43        			|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an `epoch = 15`, `batch_size = 64` and `learning_rate = 0.001`

```
from sklearn.model_selection import train_test_split

# Define training parameters
epoch = 15
batch_size = 64 
learning_rate = 0.001

# Training pipepline
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
loss = tf.reduce_mean(cross_entropy)
train_step  = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # Uses AdamOptimizer

```

* **Epochs**, I experimented with several epochs (until 50), however it stopped learning at the very early stages even with a lower learning rate, I guess due to the simplicity of the model. I decided to go with only 15 optimizing computational resources and being aware observing possible overfitting.

* **Batch size** selection is very important because it will set the pace at which your GPU will get the data. Batch size is related to the quantity of data fed to your GPU. In other words, a high batch size will result in a better estimation of the gradient when your optimizer updates the weights. However, it will be at the expense of bursting your GPU memory. That's why whenever you don't have that much of GPU resources you will be forced to reduce the batch size.

* **Learning rate** is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function. In our case, we decided that learning step to be very small at the time.

![alt text][image3]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **0.999**
* validation set accuracy of **0.956** 
* test set accuracy of **0.944**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First I tried including more dropout and more convolutional layers, but at the end in the probabilities matrix, results were very disperse. I also tried different paramenters regarding the strides and padding to filter with the convolutional layer.
* What were some problems with the initial architecture?
Output were not as expected, regarding the final class with fully connected layers
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I tunned my model as mentioned above allowing a 0.8 dropout in just one layer and using a 'VALID' padding to modify output size. At the end of the day there was a great trade off between accuracy in training/validation/test data results and chosen parameters as you can verify at the notebook.
* Which parameters were tuned? How were they adjusted and why? 
`Epochs=15`, `Learning rate=0.001` keeping `batch_size=64`. `train_step  = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # Uses AdamOptimizer`
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
As mentioned before I used 2 convolutional layers to filter the pre-proccesed input to pursue the expected output matching the classes. Droput is a great technique that helps to avoid over-fitting dropping out units (i.e.neurons) ramdomly ignoring them. This technique is just applied for training data sets, not in validation or test, forcing the neural network to learn more robust. Also I choose ReLU's as activation functions.

If a well known architecture was chosen:
* What architecture was chosen?

**LeNet** was used as base model and experimented with parameters trying to tune the parameter in the network (This part took mostly of the time for this project). Also includes 2 convolutional layers and 2 fully connected layers 

* Why did you believe it would be relevant to the traffic sign application?
This project gave the the required motivation to continue exploring and experimenting with convolutional neural networks applied to the field of Self-Driving car

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
As further improvements, I would like to experiment with other model architectures such as VGG,AlexNet, GoogleNet, etc., and possibly increase the number of CNN's.
 

### Test a Model on New Images

Performance of the model testing with images on the web (Traffic Signs), I choosed different shapes, background color, content, and different resolutions to fully test, the pipeline including the *pre-process* and the *classiffier* function.

![alt text][image4]

The performance on the new images compared to the accuracy results:

Our accuracy results were close to **~0.95** in every event (as we can observe in the notebook and above), and the performance with the new images confirm the good results of the model with the correct probabiliti in 1.0 for each tested example.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image5]

*Created by Rafael Castro*