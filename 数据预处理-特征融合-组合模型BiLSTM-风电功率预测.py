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

# smooth=savgol_filter(y,window_length,k值,mode)
# y：一维待平滑数据
# window_length：窗口长度，该值需为正奇整数
# k值：对窗口内的数据进行k阶多项式拟合，k值需要小于window_length
# mode：确定了要应用滤波器的填充信号的扩展类型

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
print(new_data.shape)


# In[3]:


from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=[-1,1])
scaler = MinMaxScaler()
new_data = scaler.fit_transform(new_data)


# In[4]:


import numpy as np

window = 15
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


# In[5]:


import tensorflow as tf
from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers
gpus=tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
tf.random.set_seed(2022)


# In[6]:


train_db = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(buffer_size = X_train.shape[0],seed = 2022).batch(batch_size = 50)
test_db = tf.data.Dataset.from_tensor_slices((X_test,y_test)).batch(batch_size = X_test.shape[0])
# print("X_train:",X_train.shape)
# print("y_train:",y_train.shape)


# In[7]:


from keras.layers import *
from keras.models import *

tf.random.set_seed(2022)
inputs = Input(shape=(15,10))

model=Conv1D(filters = 64, kernel_size = 4,strides=1,kernel_initializer=initializers.glorot_uniform(seed=2022))(inputs)#卷积层
model=MaxPooling1D(pool_size = 2)(model)#池化层
model=Conv1D(filters = 32, kernel_size = 3,strides=1,kernel_initializer=initializers.glorot_uniform(seed=2022))(model)#卷积层
model=MaxPooling1D(pool_size = 2)(model)#池化层
model=LSTM(units=80,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022))(model)#LSTM1层
model=LSTM(units=60,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022))(model)#LSTM1层

model=Flatten()(model)#flatten层
outputs = Dense(1,activation="relu",kernel_initializer=initializers.glorot_uniform(seed=2022))(model) # 全连接层
model = Model(inputs=inputs, outputs=outputs)  # 网络model
model.summary()  # 展示模型结构


# In[8]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
history = model.fit(x = train_db,validation_data = test_db,validation_freq = 1,verbose = 2,epochs = 200)


# In[9]:


model.save("Model/cnn_lstm_for_组合BiLSTM_model.h5")


# In[10]:


import json
json.dump(history.history,open("cnn_lstm_for_组合BiLSTM_history.json","w"))


# In[11]:


model_restored=tf.keras.models.load_model("Model/cnn_lstm_for_组合BiLSTM_model.h5")
pre = model_restored(X_test)
#lstm_pre=model.predict(lstm_X_test)[:,-1]
#lstm_pre=lstm_pre[:,np.newaxis]
#y_test=y_test[:,np.newaxis]
print(pre.shape)
print(y_test.shape)


# In[12]:


y_test=y_test[:,np.newaxis]
print(pre.shape)
print(y_test.shape)


# In[13]:


pre_inv=scaler.inverse_transform(np.concatenate([pre,np.ones(shape=(pre.shape[0],9))],axis=-1))[:,0]
test_inv=scaler.inverse_transform(np.concatenate([y_test.reshape(-1,1),np.ones(shape=(pre.shape[0],9))],axis=-1))[:,0]


# In[14]:


from sklearn.metrics import r2_score,mean_squared_error
print("CNN的R2",r2_score(pre_inv,test_inv))
print("CNN的MSE",mean_squared_error(pre_inv,test_inv))


# In[15]:


import numpy as np
print("原始数据的形状:",new_data.shape)
l_data=new_data[67540:,:]
print("原始数据中测试集数据的形状:",l_data.shape)
#print("pre的形状:",pre.shape)
#pre_inv=pre_inv[:,np.newaxis]

print("模型预测出来的数据形状:",pre_inv.shape)

#print("y_test的形状:",y_test.shape)


# In[16]:


pre_inv=pre_inv[:,np.newaxis]
lstm_data=np.concatenate([l_data,pre_inv],axis=-1)
print("lstm_data的形状:",lstm_data.shape)
print(lstm_data)


# In[17]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# lstm_data = scaler.fit_transform(lstm_data)


# In[18]:


import numpy as np
#print(lstm_data.shape)
window = 5
train_size = 0.75
lstm_train_index_max = int((lstm_data.shape[0] - window)*0.75) + window
lstm_X_train = []
lstm_y_train = []

lstm_X_test = []
lstm_y_test = []

# print(train_index_max)  out: 945
# print(data.shape[0]) out: 1259

for i in range(window,lstm_data.shape[0]):
    if lstm_train_index_max < i: 
        lstm_X_test.append(lstm_data[i - window:i])
        lstm_y_test.append(lstm_data[i,0])
    else:
        lstm_X_train.append(lstm_data[i - window:i])
        lstm_y_train.append(lstm_data[i,0])
lstm_X_train = np.array(lstm_X_train)
lstm_y_train = np.array(lstm_y_train)
lstm_X_test = np.array(lstm_X_test)
lstm_y_test = np.array(lstm_y_test)

print(lstm_X_train.shape,lstm_y_train.shape,lstm_X_test.shape,lstm_y_test.shape)


# In[19]:


import tensorflow as tf
from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers
gpus=tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
tf.random.set_seed(2022)


# In[20]:


lstm_train_db = tf.data.Dataset.from_tensor_slices((lstm_X_train,lstm_y_train)).shuffle(buffer_size = lstm_X_train.shape[0],seed = 2022).batch(batch_size = 50)
lstm_test_db = tf.data.Dataset.from_tensor_slices((lstm_X_test,lstm_y_test)).batch(batch_size = lstm_X_test.shape[0])
print("lstm_X_train:",lstm_X_train.shape)
print("lstm_y_train:",lstm_y_train.shape)


# In[21]:


from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers
#from AttentionLayer import AttentionLayer
from keras.layers import *
from keras.models import *

inputs = Input(shape=(5,11))

lstm_model=Bidirectional(LSTM(units=40,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022)))(inputs)#LSTM1层
lstm_model=Bidirectional(LSTM(units=30,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022)))(lstm_model)#LSTM1层
#model=LSTM(units=60,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022))(model)#LSTM1层
#mdel=Dropout(0.5)
#model=LSTM(units=124,return_sequences=True,kernel_initializer=initializers.glorot_uniform(seed=2022),recurrent_initializer=initializers.orthogonal(seed=2022))(model)#LSTM1层
attention=Dense(60, activation='sigmoid', name='attention_vec')(lstm_model)#求解Attention权重
lstm_model=Multiply()([lstm_model, attention])#attention与LSTM对应数值相乘
lstm_model=Flatten()(lstm_model)#flatten层
#model = Dense(500)(model) # 全连接层
outputs = Dense(1,activation="relu",kernel_initializer=initializers.glorot_uniform(seed=2022))(lstm_model) # 全连接层
lstm_model = Model(inputs=inputs, outputs=outputs)
lstm_model.summary()  # 展示模型结构


# In[22]:


lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
history = lstm_model.fit(x = lstm_train_db,validation_data = lstm_test_db,validation_freq = 1,verbose = 2,epochs = 150)


# In[23]:


lstm_model.save("Model/组合BiLSTM_model.h5")


# In[24]:


import json
json.dump(history.history,open("组合BiLSTM_history.json","w"))


# In[25]:


plt.rcParams['font.family'] = ['sans-serif'] # font.family设置字体样式
plt.rcParams['font.sans-serif'] = ['SimHei'] #font.sans-serif设置字体
plt.rcParams['figure.dpi'] = 600  # 图像分辨率


# In[26]:


import json
history_load=json.load(open("组合BiLSTM_history.json","r")) # json.load()：用来读取文件的，即，将文件打开然后就可以直接读取。
loss = history_load['loss'] # 训练集的每一轮的损失值
val_loss = history_load['val_loss'] # 测试集的损失值
epochs_range = range(len(loss))  # range()函数：它能返回一系列连续增加的整数
plt.plot(epochs_range, loss, label='Train Loss')  # 训练损失
plt.plot(epochs_range, val_loss, label='Test Loss')  # 测试损失
plt.legend(loc='upper right')  #设置图像的位置
plt.title('Train and Val Loss')
plt.legend() #显示标签
plt.show()


# In[27]:


model_restored=tf.keras.models.load_model("Model/组合BiLSTM_model.h5")
lstm_pre =model_restored(lstm_X_test)
#lstm_pre=model.predict(lstm_X_test)[:,-1]
#lstm_pre=lstm_pre[:,np.newaxis]
#y_test=y_test[:,np.newaxis]
print(lstm_pre.shape)
print(lstm_y_test.shape)


# In[28]:


lstm_y_test=lstm_y_test[:,np.newaxis]
print(lstm_pre.shape)
print(lstm_y_test.shape)


# In[29]:


from matplotlib import pyplot as plt
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600  # 分辨率

plt.plot(range(len(lstm_y_test)),lstm_y_test,label="real",ls="-")
plt.plot(range(len(lstm_pre)),lstm_pre,label="predicted",ls="-",lw=0.5)

plt.title("有功功率")
plt.legend()
plt.show()


# In[30]:


from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import mean_absolute_error
print("R2",r2_score(lstm_pre,lstm_y_test))
print("MSE",mean_squared_error(lstm_pre,lstm_y_test))
print("MAE",mean_absolute_error(lstm_pre,lstm_y_test))
print("RMSE",np.sqrt(mean_squared_error(lstm_pre,lstm_y_test)))


# In[ ]:




