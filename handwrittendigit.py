#importing the libraries
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.utils.np_utils import to_categorical

#loading the dataset from mnist which contains 60000 training set data and 10000 test set data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#checking the shape of training set data
print(x_train.shape)
#checking the shape of test shape data
print(x_test.shape)
#converting the data from int to float and normalizing it
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
#reshaping the data into tensor
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
#categorising the int data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#INITILISING THE CNN
model = Sequential()
#step 1-convolution step
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),activation='relu'))
#step 2-POOLING STEP
model.add(MaxPool2D(pool_size=(2,2)))
#Adding another convolution layer
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),activation='relu'))
#ading another pooling step
model.add(MaxPool2D(pool_size=(2,2)))
#step-3-FLATTENING
model.add(Flatten())
#step-4-Full Connection
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
#step-5compiling the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#summary of the model
model.summary()
#step-6 fitting the model
h=model.fit(x_train, y_train, batch_size=128,
          epochs=10, verbose=1,validation_split=0.25)
#checking the accuracy in training set by plotting graph using matplotlib
import matplotlib.pyplot as plt
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')
#evaluting the model
test_acc=model.evaluate(x_test, y_test)
#checking the test set accuracy
print(test_acc[1])#if we put 0 it will give the loss i testset and for accuracy it is 1
#our model has achieved 98.94 accuracy
