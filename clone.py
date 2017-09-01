import os
import csv
import keras
from random import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn

#Read lines from CSV and save them in a list
samples = []
with open('./data_large/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #print(line)
        samples.append(line)

#print('csv done!')

# Split the sample data to training and validation set with 80% training and 20% validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)



def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        #print('Generator shuffle complete')
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            #print('Before for loop')
            for batch_sample in batch_samples:
                name = './data_large/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                if batch_sample[3] != 'steering':
               	  center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #print('Image and angle appended')
            #print('For loop complete')
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            #print('Final for loop complete')

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(ch, row, col),output_shape=(ch, row, col)))

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
#print('Before cropping')
model.add(Cropping2D(cropping=((70,25),(0,0))))
#print('After cropping')
#print('Training start...')
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
#print('Models saved')
