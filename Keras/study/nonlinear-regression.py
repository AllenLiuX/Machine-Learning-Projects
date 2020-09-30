import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x_data = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

plt.scatter(x_data,y_data)
# plt.show()

#顺序模型
model = Sequential()
#添加全连接层
#1-10-1
# model.add(Dense(units=10, input_dim=1, activation='relu'))     # 输入维度和输出维度
model.add(Dense(units=10, input_dim=1))     # 输入维度和输出维度
model.add(Activation('tanh'))
model.add(Dense(units=1))       # 不需要设定输入维度，因为识别了上一层是10
model.add(Activation('tanh'))

# 定义优化算法
sgd = SGD(lr=0.2)

# sgd: Stochastci gradient descent：随机梯度下降
# mse: Mean Sqared Error: 均方误差
model.compile(optimizer=sgd, loss='mse')

# 训练3001个批次
for step in range(7001):
    #每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    if step%500 == 0:
        print('cost: ',cost)


#打印权值和偏置值
W, b = model.layers[0].get_weights()
print('W: ', W, 'b: ', b)

y_pred = model.predict(x_data)

plt.plot(x_data, y_pred, 'r-')
plt.show()