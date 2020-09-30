import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam

# 数据长度 一行28像素
input_size = 28
# 序列长度 28行
time_steps = 28
# 隐藏层cell个数
cell_size = 50

f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
# (x_train, y_train), (x_test, y_test) = np.load('mnist.npz')
print('x_shape:', x_train.shape) #60000
print('y_shape:', y_train.shape) #60000
# (60000, 28, 28)

x_train = x_train/255.0   #/255 -> 均一化, -1表示自动计算行数
x_test = x_test/255.0

#换one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential()
model.add(SimpleRNN(
    units=cell_size,  # output
    input_shape=(time_steps, input_size)  # input
))
# 输出层
model.add(Dense(10, activation='softmax'))

# 优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss，训练过程中计算准确率
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss', loss)
print('test accuracy', accuracy)

loss, accuracy = model.evaluate(x_train, y_train)
print('\ntrain loss', loss)
print('train accuracy', accuracy)

# 保存模型
model.save('model.h5')

# 存参数，取参数
model.save_weights('model_weights.h5')
model.load_weights('model_weights.h5')

# 保存/载入网络结构
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
