from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#Use Keras' Sequential API and provide the model
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

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# This is the augmentation configuration we will use for testing:
# Only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# This is a generator that will read pictures from the training folder
# Batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'images',  # Target Dir
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=batch_size, class_mode='binary')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=100, #Run the training epoch 100 times. The more will increase accuracy up to a certain point and will be slower. The less will run faster but decrease accuracy
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('export.h5')  #save your weights after training