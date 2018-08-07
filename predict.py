
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

#Use Keras' Sequential API and provide the model. Don't change it because .h5 file was saved containig 5 layers.
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

#Load the pretrained model weights
model.load_weights("export.h5")


img_path = 'test1/1.jpg'  # Image to be predicted


# Load the image as a tensor
new_image = load_image(img_path)

# Predict!!!
pred = model.predict(new_image) #The output is an array of 1 item. The probability.
percent = float(pred[0])*100 #Turn the array into a percentage
if pred[0] < 0.5:
    print("It's a cat!")
    print('The probability is ' + str(round(100-percent,2)) + "%")
elif pred[0] > 0.5:
    print("It's a dog!")
    print('The probability is ' + str(round(percent,2)) + "%")