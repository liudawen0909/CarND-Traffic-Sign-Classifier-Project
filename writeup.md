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

[image1]: ./img_report/CountingOfTrafficSigns.jpg "Counting_of_Traffic_Signs"
[image2]: ./img_report/grayScaling.jpg "Grayscaling"
[image4]: ./mysigns/1.png "Traffic Sign 1"
[image5]: ./mysigns/2.png "Traffic Sign 2"
[image6]: ./mysigns/3.png "Traffic Sign 3"
[image7]: ./mysigns/4.png "Traffic Sign 4"
[image8]: ./mysigns/5.png "Traffic Sign 5"
[image9]: ./mysigns/6.png "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. I have provided a Writeup that includes all the rubric points and how I addressed each one. Below is the submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/liudawen0909/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here, I plot out all the different type of signs and I also count out how many pictures we have in the training set for each kind of signs

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because compare to RGB, 
 1)grayscale is easier for data processing
 2)the features and shape of each sign are almost still the same

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it will good for the data training and increase the acurrcy of the result.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1     	| 1x1 stride, VALID padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 				|
| Convolution 2	    | 1x1 stride, VALID padding, output 10x10x16      |
| RELU     |            |
| Max polling      | 2x2 stride, VALID padding, output 5x5x16. |
| Flatten    | output 400. |
| Fully connected		| input 400, output 120    	|
| RELU     |            |
|  Fully connected		| input 120, output 84    	|
| RELU     |            |
|  Fully connected		| input 84, output 43    	|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used 30 epochs, a batch size of 100 and a learning rate of 0.0009.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

In order to get a good accuracy,
First thing I tried is applying some data preprocessing method the image data, I used gray scaling and mean value for normalized data. However the result still not good.

Then I tried to spend more effert on the image data processing. Because after I plot out some of the traffic signs picture I found that the quality of some pictures is quite low, some pictures are quite dark and some are quite bright. I go through the lessons again, I only saw there is a method dropout seems can solve this problem. However after apply the dropout method, the result seems not change so much. Maybe my implementation is wrong. Anyway, in the end I didnt choose that method.

Then I tried to increase the number of epochs, it looks better. After around 20 times training, the accurcy arrived to around 0.91. However, if I keep increase the number of epochs, I found that the value of accurcy is keep jumping.

I remember in previous class, it mentioned to tune learning rate in case of accurcy keep jumping. Then I tried to decrease the learning rate, after I set the learning rate to around 0.0009, the result looks quite good. The I choose this as the learning rate.



My final model results were:
* training set accuracy of 100%
* validation set accuracy of 93.1% 
* test set accuracy of 92.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I chose the architecture which used in the Lenet project, because this architecture is quite straight forward to me. And It also quite suitable for this project.

* What were some problems with the initial architecture?

In the training stage the accurcy keep jumping, for example in one epoch the accurcy already arrived 0.94, but in next epoch it jump back to 0.91.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

* Which parameters were tuned? How were they adjusted and why?

I decrease the learning rate to 0.0009, and increase the epochs. In this way, the accurcy will be more stable at the end of training stage. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
I choose the Lenet architecture
* Why did you believe it would be relevant to the traffic sign application?
Because in previous lesson, I learned that, it work well for numbers.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
I think for training set it should be quite high, but validation set and test set could be lower, but the accurcy value should be quite close for validation set and test set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

At the beginning, I thought use a high quality picture will give me a higher accurcy of the result. However, in the end I found it is not like that, maybe because in the training set most of the image quality is not so high. I think that could be one of the reason why I test my model on the new image the accurcy is not good. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| Ahead only     			| Ahead only 										|
| No vehicles			| Speed limit (50km/h)										|
| General caution      		| General caution				 				|
| Speed limit (30km/h)			| Speed limit (30km/h)     							|
| Go straight or left   | Go straight or left  |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 92.1%. This is lower, however, I found the issue is because in the training set all the "No vehicles" sign is tri-angle, but my test image is a circle. So, that could be the reason why the model cannot identify this traffic sign. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Bumpy road sign (96.57%), and the image does contain a Bumpy road sign. The top five soft max probabilities were

| Prediction   | Probability         	|
|:---------------------:|:---------------------------------------------:| 
| Bumpy road   | 96.57%.   |
| General caution | 2.28%. |
| Dangerous curve to the right | 0.56%. |
| Road work | 0.26%. |
| Bicycles crossing  | 0.25%. |

For the second image

| Prediction	 | Probability  |
|:---------------------:|:---------------------------------------------:| 
| Ahead only  | 100.00% |
| Road work   |  0.00% |
| Turn right ahead |  0.00%  |
| Bicycles crossing  | 0.00%  |
| Go straight or right | 0.00%  |

For the third image

|     Prediction	        			| Probability         	|
|:---------------------:|:---------------------------------------------:| 
|Speed limit (50km/h)| 97.36%|
|Speed limit (30km/h)| 2.45%|
|No vehicles| 0.15%|
|Keep right| 0.03%|
|Speed limit (70km/h)| 0.00%|

For the fourth image

| Prediction	 	| Probability  	|
|:---------------------:|:---------------------------------------------:| 
| General caution | 100.00% |
| Traffic signals | 0.00% |
| Pedestrians | 0.00% |
| Right-of-way at the next intersection| 0.00% |
| Road work | 0.00% |

For the fifth image

| Prediction	        			| Probability         	|
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) | 100.00% |
| Speed limit (50km/h) | 0.00% |
| Right-of-way at the next intersection | 0.00% |
| Roundabout mandatory | 0.00% |
| Speed limit (60km/h) | 0.00% |

For the sixth image

| Prediction	| Probability |
|:---------------------:|:---------------------------------------------:| 
| Go straight or left | 100.00% |
| Dangerous curve to the right | 0.00% |
| Roundabout mandatory | 0.00% |
| No passing for vehicles over 3.5 metric tons | 0.00% |
| Keep right | 0.00% |

The third image actually should be no vehicles, but in prediction it shows Speed limit (50km/h), means the model still have problem need to speed more effect on the data processing and achitecture chosen.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


