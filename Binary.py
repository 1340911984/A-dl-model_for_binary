# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras
import numpy as np
# Part 1 - Building the CNN
import matplotlib.pyplot as plt
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, AveragePooling2D, Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# Step 2 - Pooling
classifier.add(MaxPooling2D((2, 2)))
# Adding 2,3,4 convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(AveragePooling2D(2,2))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(AveragePooling2D(2,2))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dropout(0.5))
classifier.add(Dense(512, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:\\数据集\\data_set\\data\\train',
                                                 target_size = (150, 150),
                                                 batch_size = 256,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('D:\\数据集\\data_set\\data\\val',
                                            target_size = (150, 150),
                                            batch_size =16,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 191,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 473)

res = classifier.evaluate(test_set,batch_size=25)  # 返回损失和精度
#最后，使用evaluate方法来评估模型
#到这一步，我们创建了模型
# 接下来就是调用fit函数将数据提供给模型
print(classifier.metrics_names);'''。
这里还可以指定批次大小（batch size）、迭代次数、验证数据集等等。
其中批次大小、迭代次数需要根据数据规模来确定，并没有一个固定的最优值。'''
print(res)

prob = classifier.predict(test_set)
print(np.argmax(prob))
print(np.argmax(prob))
print(np.argmax(prob))
