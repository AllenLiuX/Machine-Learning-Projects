import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

x_data = np.random.rand(100)
noise = np.random.normal(0,0.01, x_data.shape)  #center, dv, size
y_data = x_data*0.1 + 0.2 + noise
# print(noise)

plt.scatter(x_data, y_data)
# plt.show()

#顺序模型
model = Sequential()
#添加全连接层
model.add(Dense(units=1, input_dim=1))
#sgd: Stochastci gradient descent：随机梯度下降
#mse: Mean Sqared Error: 均方误差
model.compile(optimizer='sgd', loss='mse')

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