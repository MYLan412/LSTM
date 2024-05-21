#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=pd.read_excel("./美国数据.xlsx")
print(data.info())#.info()函数：给出样本数据的相关信息概览 ：行数，列数，列索引，列非空值个数，列类型，内存占用


# In[2]:


d=data.values[:401,3:]#.values():查看字典中元素的函数，返回值都为一个list(列表）
print(d.shape)#.shape:读取矩阵的长度，582行，4列


# In[3]:


from sklearn.preprocessing import MinMaxScaler  #sklearn的归一化方法MinMaxScaler
scaler=MinMaxScaler()  #初始化一个MinMaxScaler对象，将数据映射到[0，1]区间
d=scaler.fit_transform(d)  #拟合并转换数据，本质上就是先求最大最小值，然后对数据按照公式计算，即自动归一化
window=10
X=[]
y=[]

for i in range(window,len(d)):  #给i进行赋值
    X.append(d[i-window:i,:][np.newaxis,:,:])  #.append()函数：用于在X列表末尾添加新的对象
    y.append(d[i])
X=np.concatenate(X,axis=0)  #.concatenate()函数：将多个数组进行拼接
y=np.array(y)
print(X.shape,y.shape)
print(y)


# In[4]:


train_size=0.75
X_train,X_test=X[:int(len(X)*0.75),:,:],X[int(len(X)*0.75):,:,:]  #[a:b,c:d]分析时以逗号为分隔符，行取a到b-1，列取c到d-1
y_train,y_test=y[:int(len(X)*0.75),:],y[int(len(X)*0.75):,:]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score  # R square
# from math import sqrt
from keras.layers import *
from keras.models import *
# from keras.optimizers import Adam
# import keras.backend as K
from timeit import default_timer as timer
import tensorflow as tf
# from AttentionLayer import MyAttention
# from test import Attention_LSTM_Cell_2

gpus = tf.config.experimental.list_physical_devices('GPU') #可以获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  #作用是限制显存的使用


# In[6]:


from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers

tf.random.set_seed(2021) # 全局设置tf.set_random_seed()函数，使用之后后面设置的随机数都不需要设置seed，而可以跨会话生成相同的随机数。
inputs = Input(shape=(window, X_train.shape[2]))  # 输入形状(window,input_size)

#model = (GRU(15,return_sequences=False))(inputs)
model=LSTM(units=10,return_sequences=True)(inputs)
#model=Dropout(0.3)(model)

model=Flatten()(model)  #flatten()：Flatten层用来将输入压平，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
outputs = Dense(3)(model)  # 全连接层
model = Model(inputs=inputs, outputs=outputs)  # 网络model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))  # 损失函数，优化器
model.summary()  # 展示模型结构
start = timer()
history = model.fit(X_train, y_train, batch_size=60, epochs=200, shuffle=False, validation_data=(X_test, y_test),
                    validation_freq=1)  # 拟合神经网络，训练模型epoch次

end = timer()
print(end - start)
#


# In[7]:


model.save("Model/lstm_AC19_model.h5") # 该方法能够将整个模型进行保存：模型结构，能够重新实例化模型；模型权重；优化器的状态，在上次中断的地方继续训练；
#


# In[8]:


import json
json.dump(history.history,open("lstm_AC19_history.json","w")) # 将json格式的数据写进文件里


# In[9]:


plt.rcParams['font.family'] = ['sans-serif'] # font.family设置字体样式
plt.rcParams['font.sans-serif'] = ['SimHei'] #font.sans-serif设置字体
plt.rcParams['figure.dpi'] = 600  # 图像分辨率


# In[10]:


import json
history_load=json.load(open("lstm_AC19_history.json","r")) # json.load()：用来读取文件的，即，将文件打开然后就可以直接读取。
loss = history_load['loss'] # 训练集的每一轮的损失值
val_loss = history_load['val_loss'] # 测试集的损失值
epochs_range = range(len(loss))  # range()函数：它能返回一系列连续增加的整数
plt.plot(epochs_range, loss, label='Train Loss')  # 训练损失
plt.plot(epochs_range, val_loss, label='Test Loss')  # 测试损失
plt.legend(loc='upper right')  #设置图像的位置
plt.title('Train and Val Loss')
plt.legend() #显示标签
plt.show()


# In[11]:


def mape(y_true, y_pred):  # mape:平均绝对百分比误差
    y_t_index = y_true != 0
    y_t = y_true[y_t_index] # 测试集目标真实值
    y_p = y_pred[y_t_index] # 测试集目标预测值
    return np.mean(np.abs((y_p - y_t) / y_t))


# In[12]:


model_restored=tf.keras.models.load_model("Model/lstm_AC19_model.h5") # 重新实例化保存的模型，通过该方法返回的模型是已经编译过的模型
 # 当使用predict()方法进行预测时，返回值是数值，表示样本属于每一个类别的概率
 # scaler.inverse_transform()：将标准化后的数据转换为原始数据
test_pre=scaler.inverse_transform(model_restored.predict(X_test)) # 预测数据反归一化
test_true=scaler.inverse_transform(y_test) # 真实数据反归一化


# In[13]:


test_pre_t1=test_pre[:,0]
test_true_t1=test_true[:,0]
print('测试集上确诊人数的MAE/RMSE/MAPE/R square')
print(mean_absolute_error(test_true_t1, test_pre_t1))
print(np.sqrt(mean_squared_error(test_true_t1, test_pre_t1)))
print(mape(test_true_t1, test_pre_t1))
print(r2_score(test_true_t1, test_pre_t1))

test_pre_t2=test_pre[:,1]
test_true_t2=test_true[:,1]
print('测试集上治愈人数的MAE/RMSE/MAPE/R square')
print(mean_absolute_error(test_true_t2, test_pre_t2))
print(np.sqrt(mean_squared_error(test_true_t2, test_pre_t2)))
print(mape(test_true_t2, test_pre_t2))
print(r2_score(test_true_t2, test_pre_t2))

test_pre_t3=test_pre[:,2]
test_true_t3=test_true[:,2]
print('测试集上死亡人数的MAE/RMSE/MAPE/R square')
print(mean_absolute_error(test_true_t3, test_pre_t3))
print(np.sqrt(mean_squared_error(test_true_t3, test_pre_t3)))
print(mape(test_true_t3, test_pre_t3))
print(r2_score(test_true_t3, test_pre_t3))


# In[ ]:




