#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=pd.read_excel("./中国数据.xlsx")
print(data.info())#.info()函数：给出样本数据的相关信息概览 ：行数，列数，列索引，列非空值个数，列类型，内存占用


# In[2]:


d=data.values[112:,1:]#.values():查看字典中元素的函数，返回值都为一个list(列表）
print(d.shape)#.shape:读取矩阵的长度，582行，4列


# In[3]:


from sklearn.preprocessing import MinMaxScaler  #sklearn的归一化方法MinMaxScaler
scaler=MinMaxScaler()  #初始化一个MinMaxScaler对象，将数据映射到[0，1]区间
d=scaler.fit_transform(d)  #拟合并转换数据，本质上就是先求最大最小值，然后对数据按照公式计算，即自动归一化

window = 10
X_train = []
y_train = []

X_test = []
y_test = []

for i in range(window,d.shape[0]):
    X_test.append(d[i - window:i])
    y_test.append(d[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_test.shape,y_test.shape)


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score  # R square
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers

gpus = tf.config.experimental.list_physical_devices('GPU') #可以获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  #作用是限制显存的使用


# In[5]:


def mape(y_true, y_pred):  # mape:平均绝对百分比误差
    y_t_index = y_true != 0
    y_t = y_true[y_t_index] # 测试集目标真实值
    y_p = y_pred[y_t_index] # 测试集目标预测值
    return np.mean(np.abs((y_p - y_t) / y_t))


# In[6]:


model_restored=tf.keras.models.load_model("Model/cnn_woa_lstm_attention_model.h5") # 重新实例化保存的模型，通过该方法返回的模型是已经编译过的模型
 # 当使用predict()方法进行预测时，返回值是数值，表示样本属于每一个类别的概率
 # scaler.inverse_transform()：将标准化后的数据转换为原始数据
test_pre=scaler.inverse_transform(model_restored(X_test)) # 预测数据反归一化
test_true=scaler.inverse_transform(y_test) # 真实数据反归一化

#print(model_restored(X_test))
#print(model_restored.predict(X_test))


# In[7]:


from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600  # 分辨率

test_pre_t1=test_pre[:,0]
test_true_t1=test_true[:,0]
print('测试集上累计确诊人数的MAE/RMSE/MAPE/R square')
print(mean_absolute_error(test_true_t1, test_pre_t1))
print(np.sqrt(mean_squared_error(test_true_t1, test_pre_t1)))
print(mape(test_true_t1, test_pre_t1))
print(r2_score(test_true_t1, test_pre_t1))

test_pre_t2=test_pre[:,1]
test_true_t2=test_true[:,1]
print('测试集上累计治愈人数的MAE/RMSE/MAPE/R square')
print(mean_absolute_error(test_true_t2, test_pre_t2))
print(np.sqrt(mean_squared_error(test_true_t2, test_pre_t2)))
print(mape(test_true_t2, test_pre_t2))
print(r2_score(test_true_t2, test_pre_t2))

test_pre_t3=test_pre[:,2]
test_true_t3=test_true[:,2]
print('测试集上累计死亡人数的MAE/RMSE/MAPE/R square')
print(mean_absolute_error(test_true_t3, test_pre_t3))
print(np.sqrt(mean_squared_error(test_true_t3, test_pre_t3)))
print(mape(test_true_t3, test_pre_t3))
print(r2_score(test_true_t3, test_pre_t3))

# test_pre_t4=test_pre[:,3]
# test_true_t4=test_true[:,3]
# print('测试集上新增治愈人数的MAE/RMSE/MAPE/R square')
# print(mean_absolute_error(test_true_t4, test_pre_t4))
# print(np.sqrt(mean_squared_error(test_true_t4, test_pre_t4)))
# print(mape(test_true_t4, test_pre_t4))
# print(r2_score(test_true_t4, test_pre_t4))

# test_pre_t5=test_pre[:,4]
# test_true_t5=test_true[:,4]
# print('测试集上累计死亡人数的MAE/RMSE/MAPE/R square')
# print(mean_absolute_error(test_true_t5, test_pre_t5))
# print(np.sqrt(mean_squared_error(test_true_t5, test_pre_t5)))
# print(mape(test_true_t5, test_pre_t5))
# print(r2_score(test_true_t5, test_pre_t5))

# test_pre_t6=test_pre[:,5]
# test_true_t6=test_true[:,5]
# print('测试集上新增死亡人数的MAE/RMSE/MAPE/R square')
# print(mean_absolute_error(test_true_t6, test_pre_t6))
# print(np.sqrt(mean_squared_error(test_true_t6, test_pre_t6)))
# print(mape(test_true_t6, test_pre_t6))
# print(r2_score(test_true_t6, test_pre_t6))


# In[ ]:





# In[ ]:




