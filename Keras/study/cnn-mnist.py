import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import pydot
import graphviz

# from keras.datasets import mnist

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
# x_train = x_train.reshape(x_train.shape[0], -1)/255.0   #/255 -> 均一化, -1表示自动计算行数
# x_test = x_test.reshape(x_test.shape[0], -1)/255.0

x_train = x_train.reshape(-1, 28, 28, 1)/255.0   #/255 -> 均一化, -1表示自动计算行数
x_test = x_test.reshape(-1, 28, 28, 1)/255.0

#换one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential()
model.add(Convolution2D(
    input_shape=(28, 28, 1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))
# 第一个池化层
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
))
# 第二个卷积层
model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
# 第二个池化层
model.add(MaxPooling2D(2,2,'same'))
# 扁平化为1维
model.add(Flatten())
# 第一个全连接层
model.add(Dense(1024, activation='relu'))
# Dropout
model.add(Dropout(0.5))
# 第二个全连接层
model.add(Dense(10, activation='softmax'))
# 优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss，训练过程中计算准确率
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=128, epochs=0)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss', loss)
print('test accuracy', accuracy)
#128, 4 -> 98.61%, 98.66%

loss, accuracy = model.evaluate(x_train, y_train)
print('\ntrain loss', loss)
print('train accuracy', accuracy)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
plt.figure(figsize=(10,10))
img = plt.imread('model.png')
plt.imshow(img)
plt.axis('off')
plt.show()
