# **Behavioral Cloning**

## **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image5]: ./examples/learning_curve.png "Learning curve"
[image4]: ./examples/steering_histo_before.png  "Steering histo before"
[image6]: ./examples/steering_histo_after.png "Steering histo after"
[image7]: ./examples/img_flipped.png "Flipped"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* log_loader.py containing the script to load data from driving log and filter them
* preprocess_data.py containing the script to preprocess images and augument training and validation data
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md writup, you ar ereading now

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
In this file I provided _Nvidia_Model_ class that implements NVIDIA CNN architecture : creates, trains and saves model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model basically (with some small modification) corresponds to NVIDIA model used in the project too.
It consists four convolution layers and four fully connected layers.

I provide some modification to the model:
* reduced the input images size tp 64x64x3 as suggested in [help guide](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) . It speed up model training and did not have major impact on the accuracy.
* normalized the images outside of the model. Providing of filtering, pre-processing and augmentation of data is too complex to do it in the model (lambda layer)
* used Exponential Linear Units (ELU) as activation function as suggested in [paper](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)

#### 2. Attempts to reduce overfitting in the model

The model doesn't contain dropout layers. As explained in the projected an augmentation of data set helps to avoid overfitting.
I tried to introduce dropout layers but the experience showed me that they have bad influence on my model.
The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I didn't use the data set provide by Udacity. I decided to create my own data.
For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I established a solid data loading and processing pipeline to achieve a big training set of data I decided to omit a step for providing a very simple model and use the [NIVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model directly.

As the model worked pretty good I didn't change it significantly.

I used an _adam_ optimizer with default parameters and  choosed loss function was mean squared error (MSE).
I used 512 batch size and 5 epochs.

The learning curve showed a mean squared error less then 0.02 :

![alt text][image5]

The final step was to run the simulator to see how well the car was driving around "Lake" test track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer 				| Description			 						|
|:---------------------:|:---------------------------------------------:|
| Input         		| 64x64x3 RGB image   							|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 60x60x24 	|
| ELu					|												|
| Max pooling			| 2x2 stride, outputs 30x30x24 				    |
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 26x26x36	|
| ELu					|												|
| Max pooling			| 2x2 stride, outputs 13x13x36					|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 9x9x48		|
| ELu					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x48					|
| Convolution 3x3		| 1x1 stride, VALID padding, outputs 3x3x64		|
| ELu					|												|
| Convolution 3x3		| 1x1 stride, SAME padding, outputs 3x3x64		|
| ELu					|												|
| Flatten   			| outputs 3x3x64 = 576		        			|
| Fully connected		| outputs 576      						    	|
| ELu					|												|
| Fully connected		| outputs 100      								|
| ELu					|												|
| Fully connected		| outputs 50  									|
| ELu					|												|
| Fully connected		| outputs 10  									|
| ELu					|												|
| Logits				| outputs 1     								|

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer 				| Output Shape              | Param #   	|
|:---------------------:|:--------------------------|--------------:|
| conv2d_1 (Conv2D)            |(None, 60, 60, 24)  |      1824     |
| max_pooling2d_1 (MaxPooling2 |(None, 30, 30, 24)  |      0        |
| conv2d_2 (Conv2D)            |(None, 26, 26, 36)  |      21636    |
| max_pooling2d_2 (MaxPooling2 |(None, 13, 13, 36)  |      0        |
| conv2d_3 (Conv2D)            |(None, 9, 9, 48)    |      43248   |
| max_pooling2d_3 (MaxPooling2 |(None, 5, 5, 48)    |      0        |
| conv2d_4 (Conv2D)            |(None, 3, 3, 64)    |      27712     |
| conv2d_5 (Conv2D)            |(None, 3, 3, 64)    |      36928    |
| flatten_1 (Flatten)          |(None, 576)         |      0        |
| dense_1 (Dense)              |(None, 576)         |      332352   |
| dense_2 (Dense)              |(None, 100)         |      57700    |
| dense_3 (Dense)              |(None, 50)          |      5050     |
| dense_4 (Dense)              |(None, 10)          |      510      |
| dense_5 (Dense)              |(None, 1)           |      11       |

#### 3. Creation of the Training Set & Training Process

##### Collecting training data
I used "Lake" track for training and validation. I capture 3 laps of this track : two in clockwise and one in counter-clockwise direction.

Driving in counter-clockwise direction to avoid that data will be biased towards left turns and to the model a new track to learn from, so the model will generalize better.
I tried to keep car at the center of lane.

The capture of data from recovery driving from the sides was unnecessary.

##### Data pre-processing
This is a very important step for training process. Following steps were done:
* image was cropped from top and bottom as not all of these pixels contain useful information. The top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car.
* image was resized to 64x64x3.
* image normalization (YUV conversion)

Important: as this pre-processing step was used by data augmentation it is neceessary  to provide it in drive.py file as the images provided from simulator have different size.

I used generators to stream training and validation data from disk instead of storing the preprocessed data in memory all at once what  is much more memory-efficient.

##### Data augmentation

To augment the data set :
* I used the images from the left and right camera. I adjusted steering values about constant correction  +/- 0.2 respectively.
* I flipped images horizontaly and put opposite sign of the steering measurement. This seems to be an effective technique for helping with the left turn bias and makes sense for my test drive in counter-clockwise direction.

In this way the training set increased 6 times.

For example, here is an image that has then been flipped:

![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by ...

##### Filtering input data
The "Lake" test track includes long sections with very slight or no curvature, the data captured from it tends to be heavily skewed toward low and zero turning angles. This creates a problem for the neural network, which then becomes biased toward driving in a straight line and can become easily confused by sharp turns.

The distribution of the input data in respect to steering angle can be observed in bar graph below:

![alt text][image4]

To reduce the occurrence of low and zero angle data points, I created a histogram with 1° steering angle resolution using numpy.histogram. Using a rule that every history bin shall have not more than 2^0.5 (1.414) times entries than the mean count of all filled bins I could achieve a better distribution of steering angles. Based on this histogram - see bar graph below -I  calculated keeping probabilities that I used to filter the data.

![alttext][image6]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
