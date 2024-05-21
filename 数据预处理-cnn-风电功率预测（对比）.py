#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_excel("风电功率数据.xlsx")
print(data.info())

data.pop("记录的天数")
data.pop("数据的采集时间")
data.pop("风力发电机组ID")
print(data.info())


# In[2]:


from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import json

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 600  # 分辨率

f0=data.values[:,0]
ff0=savgol_filter(f0,23,19,mode='nearest')
ff0=ff0[:,np.newaxis]
# print(ff0.shape)
f1=data.values[:,1]
ff1=savgol_filter(f1,23,19,mode='nearest')
ff1=ff1[:,np.newaxis]

f2=data.values[:,2]
ff2=savgol_filter(f2,23,19,mode='nearest')
ff2=ff2[:,np.newaxis]
print(ff2)

f3=data.values[:,3]
ff3=savgol_filter(f3,19,17,mode='nearest')
ff3=ff3[:,np.newaxis]

f4=data.values[:,4]
ff4=savgol_filter(f4,19,17,mode='nearest')
ff4=ff4[:,np.newaxis]

f5=data.values[:,5]
ff5=savgol_filter(f5,19,17,mode='nearest')
ff5=ff5[:,np.newaxis]

f6=data.values[:,6]
ff6=savgol_filter(f6,19,17,mode='nearest')
ff6=ff6[:,np.newaxis]

f7=data.values[:,7]
ff7=savgol_filter(f7,19,17,mode='nearest')
ff7=ff7[:,np.newaxis]

f8=data.values[:,8]
ff8=savgol_filter(f8,19,17,mode='nearest')
ff8=ff8[:,np.newaxis]

f9=data.values[:,9]
ff9=savgol_filter(f9,23,19,mode='nearest')
ff9=ff9[:,np.newaxis]

new_data=np.concatenate([ff0,ff1,ff2,ff3,ff4,ff5,ff6,ff7,ff8,ff9],axis=-1)
new_data=new_data[67540:,:]
print(new_data.shape)


# In[3]:


from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=[-1,1])
scaler = MinMaxScaler()
new_data = scaler.fit_transform(new_data)

window = 5
train_size = 0.75
train_index_max = int((new_data.shape[0] - window)*0.75) + window
X_train = []
y_train = []

X_test = []
y_test = []

# print(train_index_max)  out: 945
# print(data.shape[0]) out: 1259

for i in range(window,new_data.shape[0]):
    if train_index_max < i: 
        X_test.append(new_data[i - window:i])
        y_test.append(new_data[i,0])
    else:
        X_train.append(new_data[i - window:i])
        y_train.append(new_data[i,0])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[4]:


import tensorflow as tf
from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers
# from AttentionLayer import AttentionLayer
gpus=tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
tf.random.set_seed(2022)


# In[5]:


train_db = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(buffer_size = X_train.shape[0],seed = 2022).batch(batch_size = 50)
test_db = tf.data.Dataset.from_tensor_slices((X_test,y_test)).batch(batch_size = X_test.shape[0])
print("X_train:",X_train.shape)
print("y_train:",y_train.shape)


# In[6]:


from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers
# from AttentionLayer import AttentionLayer
from keras.layers import *
from keras.models import *

model = Sequential()
inputs = Input(shape=(5,10))

model=Conv1D(filters = 64, kernel_size = 4,strides=1,kernel_initializer=initializers.glorot_uniform(seed=2022))(inputs)#卷积层
model=MaxPooling1D(pool_size = 2)(model)#池化层
# model=LSTM(units=80,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022))(inputs)#LSTM1层
# model=LSTM(units=60,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022))(model)#LSTM1层
# attention=Dense(60, activation='sigmoid', name='attention_vec')(model)#求解Attention权重
# model=Multiply()([model, attention])#attention与LSTM对应数值相乘
model=Flatten()(model)#flatten层
outputs = Dense(1,activation="relu",kernel_initializer=initializers.glorot_uniform(seed=2022))(model) # 全连接层

model = Model(inputs=inputs, outputs=outputs)
model.summary()  # 展示模型结构


# In[7]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
history = model.fit(x = train_db,validation_data = test_db,validation_freq = 1,verbose = 2,epochs = 150)


# In[8]:


model.save("Model/cnn_model.h5")


# In[9]:


import json
json.dump(history.history,open("cnn_history.json","w"))


# In[10]:


model_restored=tf.keras.models.load_model("Model/cnn_model.h5")
pre =model_restored(X_test)
#lstm_pre=model.predict(lstm_X_test)[:,-1]
#lstm_pre=lstm_pre[:,np.newaxis]
#y_test=y_test[:,np.newaxis]
print(pre.shape)
print(y_test.shape)


# In[11]:


y_test=y_test[:,np.newaxis]
print(pre)
print(y_test)


# In[12]:


from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import mean_absolute_error
print("R2",r2_score(pre,y_test))
print("MSE",mean_squared_error(pre,y_test))
print("MAE",mean_absolute_error(pre,y_test))
print("RMSE",np.sqrt(mean_squared_error(pre,y_test)))


# In[ ]:




