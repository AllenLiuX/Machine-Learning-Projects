import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.models import load_model

f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()


x_train = x_train/255.0   #/255 -> 均一化, -1表示自动计算行数
x_test = x_test/255.0

#换one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = load_model('model.h5')

loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss', loss)
print('accuracy', accuracy)

# retrain
model.fit(x_train, y_train, batch_size=64, epochs=3)
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss', loss)
print('accuracy', accuracy)

