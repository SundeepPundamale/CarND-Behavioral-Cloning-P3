# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/recovery1.jpg "Recovery Image"
[image8]: ./examples/recovery2.jpg "Recovery Image"
[image9]: ./examples/recovery3.jpg "Recovery Image"
[image10]: ./examples/recovery4.jpg "Recovery Image"
[image11]: ./examples/center1.jpg "Center Image"
[image12]: ./examples/center2.jpg "Center Image"
[image13]: ./examples/left1.jpg "Left Image"
[image14]: ./examples/right1.jpg "Right Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My architecture is similar to the [NVIDIA Network architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) as published in their End-to-end Deep learning self driving car publication. I also viewed [David's Q&A](https://www.youtube.com/watch?v=rpxZ87YFg0M&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&index=3) on youtube and used the generators boiler plate from the udacity [generators lesson](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/b602658e-8a68-44e5-9f0b-dfa746a0cc1a). I used the aforementioned references as a starting point to implement my model.

My models looked as follows:

| Layer         		|     Description	        					      | 
|:---------------------:|:---------------------------------------------------:|   
| normalisation         | x/255.0 - 0.5                                       |
| convolution           | 5x5, filter: 24, subsampling: 2x2, activation: RELU | 
| convolution           | 5x5, filter: 36, subsamplng: 2x2, activation: RELU  |
| Convolution           | 5x5, filter: 48, subsampling: 2x2, activation: RELU |
| Convolution           | 3x3, filter: 64, activation: RELU                   | 
| Convolution           | 3x3, filter: 64, activation: RELU                   |
| Flatten               | Flatten Layer                                       |
| Fully connected       | neurons: 100                                        | 
| Fully connected       | neurons: 50                                         |
| Fully connected       | neurons: 10                                         |
| Fully connected       | neurons: 1                                          |

#### 2. Attempts to reduce overfitting in the model

I did not use any dropout layers. However, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 2 laps of center lane driving and 3rd lap for recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

#### 1. Solution Design Approach

I build my model based on the [NVIDIA Network architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)


I experimented by running the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, on the first try the car went off the track around sharp curves and was not able to recover. Then i cropped the image such that sky, trees and the car hood was excluded. In my next attempt the car still fell off track. At this point i had 1 lap of center lane driving as my training data. I recorded 2 laps of center lane driving next. This time the car could not cross the bridge. In my next try i recorded recovering the car from outside the lane. I focussed recovering data around the bridge. But the car still fell of the track in sharp curves. I finally recorded the recovery from both left and right side of the lanes near the sharp curves and at the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 61-86) consisted of a normalizer, 3 convolution layers with a 2x2 stride and a 5x5 kernel, 2 convolution layers without a stride and 3x3 kernel size, followed by a flattening layer and 4 fully connected layers.  

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image12]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when crossing the lane. These images show what a recovery looks like starting from left side and right side of the lane :

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

Below are the images from left and right cameras:

![alt text][image13]
![alt text][image14]


After the collection process, I had 8832 number of data points. I  preprocessed this data by normalising and cropping it. I also randomly shuffled the data set and put 20% of the data into a validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
