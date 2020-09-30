import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
# (x_train, y_train), (x_test, y_test) = np.load('mnist.npz')
# (60000, 28, 28)
print('x_shape:', x_train.shape) #60000
print('y_shape:', y_train.shape) #60000
# 60000, 28, 28 -> 60000, 784
x_train = x_train.reshape(x_train.shape[0], -1)/255.0   #/255 -> 均一化, -1表示自动计算行数
x_test = x_test.reshape(x_test.shape[0], -1)/255.0

#换one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential()
model.add(Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax'))

sgd = SGD(lr=0.2)

# 定义优化器，loss，训练过程中计算准确率
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss', loss)
print('accuracy', accuracy)
