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

window = 10
train_size = 0.75
train_index_max = int((d.shape[0] - window)*0.75) + window
X_train = []
y_train = []

X_test = []
y_test = []

for i in range(window,d.shape[0]):
    if train_index_max < i: 
        X_test.append(d[i - window:i])
        y_test.append(d[i])
    else:
        X_train.append(d[i - window:i])
        y_train.append(d[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score  # R square
# from math import sqrt
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
# from keras.optimizers import Adam
# import keras.backend as K
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.keras import layers,Sequential,losses,optimizers,metrics,models,initializers
#from AttentionLayer import AttentionLayer
# from AttentionLayer import MyAttention
# from test import Attention_LSTM_Cell_2

gpus = tf.config.experimental.list_physical_devices('GPU') #可以获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  #作用是限制显存的使用


# In[5]:


import math
from keras.layers import *
from keras.models import *
# 定义适应度函数，用来计算模型的误差值
# 输入待优化的参数、训练集中的输入、训练集中的输出、测试集中的输入、测试集中的输出
def fitness(pop,P,T,Pt,Tt):    
    tf.random.set_seed(2021)
    lr = pop[0]  # 学习率
    num_epochs = int(pop[1])  # 迭代次数
    batch_size = int(pop[2])  # batch_size
    hidden1 = int(pop[3])  # lstm1隐藏层节点数
    hidden2 = int(pop[4])  # stm2隐藏层节点数

    window, feature = P.shape[-2:]
    
    inputs = Input(shape=(window, feature))  # 输入形状(window,input_size)
    model=Conv1D(filters = 7, kernel_size = 2,strides=1,kernel_initializer=initializers.glorot_uniform(seed=2022))(inputs)#卷积层
    model=MaxPooling1D(pool_size = 3,strides=2)(model)#池化层
    model=LSTM(units=hidden1,return_sequences=True)(model)#LSTM1
    model=LSTM(units=hidden2,return_sequences=True)(model)#LSTM2
    #model=Dropout(0.3)(model)
    attention=Dense(hidden2, activation='sigmoid', name='attention_vec')(model)#求解Attention权重
    model=Multiply()([model, attention])#attention与LSTM对应数值相乘
    model=Flatten()(model)  #flatten()：Flatten层用来将输入压平，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    outputs = Dense(3,kernel_initializer=initializers.glorot_uniform(seed=2022))(model)  # 全连接层
    model = Model(inputs=inputs, outputs=outputs)  # 网络model
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')# 定义优化器
    
    model.fit(P, T,epochs=num_epochs,batch_size=batch_size, verbose=0)# 训练
    
    Tp = model.predict(Pt)# 得到模型预测值
        
    F2=np.mean(np.square((Tp-Tt)))# 计算模型误差
    return F2

# 鲸鱼优化算法
def WOA(p_train,t_trian,p_test,t_test):

    max_iterations=10# 最大迭代次数
    noposs=5# 种群数量
   
    lb=[0.001,10, 16, 30 ,1 ]
    ub=[0.01, 500,128,50,30]# 寻优上下界
    noclus=len(lb)# 寻优维度

    poss_sols = np.zeros((noposs, noclus)) # 鲸鱼位置
    gbest = np.zeros((noclus,)) # 全局最优鲸鱼位置
    b = 2.0
    
   # 种群初始化
    for i in range(noposs):
        for j in range(noclus):
            if j==0:
                poss_sols[i][j] =(ub[j]-lb[j])*np.random.rand()+lb[j]
            else:
                poss_sols[i][j] =  np.random.randint(lb[j],ub[j])

    global_fitness = np.inf
    for i in range(noposs):
        cur_par_fitness = fitness(poss_sols[i,:],p_train,t_trian,p_test,t_test) 
        if cur_par_fitness < global_fitness:
            global_fitness = cur_par_fitness
            gbest = poss_sols[i].copy()
            
    # 开始迭代
    trace,trace_pop=[],[]
    for it in range(max_iterations):
        for i in range(noposs):
            # 定义计算时所需的参数
            a = 2.0 - (2.0*it)/(1.0 * max_iterations)
            r = np.random.random_sample()
            A = 2.0*a*r - a
            C = 2.0*r
            l = 2.0 * np.random.random_sample() - 1.0
            p = np.random.random_sample()
            
            for j in range(noclus):
                x = poss_sols[i][j]
                # 收缩包围机制 P<0.5
                if p < 0.5:
                    if abs(A) < 1:
                        _x = gbest[j].copy()
                    else :
                        rand = np.random.randint(noposs)
                        _x = poss_sols[rand][j]
                    D = abs(C*_x - x)
                    updatedx = _x - A*D
                # 螺旋更新 P>=0.5
                else :
                    _x = gbest[j].copy()
                    D = abs(_x - x)
                    updatedx = D * math.exp(b*l) * math.cos(2.0* math.acos(-1.0) * l) + _x
                
                poss_sols[i][j] = updatedx# 鲸鱼位置更新
                
            poss_sols[i,:]=boundary(poss_sols[i,:],lb,ub)#边界判断
            
            fitnessi = fitness(poss_sols[i],p_train,t_trian,p_test,t_test)# 计算鲸鱼的适应度函数
            if fitnessi < global_fitness :
                global_fitness = fitnessi
                gbest = poss_sols[i].copy()
        trace.append(global_fitness)
        print ("iteration",it+1,"=",global_fitness,[gbest[i] if i==0 else int(gbest[i]) for i in range(len(lb))])

        trace_pop.append(gbest)
    return gbest, trace,trace_pop

def boundary(pop,lb,ub):

    pop=[pop[i] if i==0 else int(pop[i]) for i in range(len(lb))]
    for j in range(len(lb)):
        if pop[j]>ub[j] or pop[j]<lb[j]:
            if j==0:
                pop[j] =(ub[j]-lb[j])*np.random.rand()+lb[j]
            else:
                pop[j] =  np.random.randint(lb[j],ub[j])
    return pop

# 开始优化
best,trace,process=WOA(X_train,y_train,X_test,y_test)
trace,process=np.array(trace),np.array(process)
np.savez('Model/woa_result.npz',trace=trace,best=best,process=process)


# In[6]:


def HUATU(trace,result):
    plt.figure()
    plt.plot(trace)
    plt.title('fitness curve')
    plt.xlabel('iteration')
    plt.ylabel('fitness value')
    
    plt.figure()
    plt.plot(result[:,0])
    plt.xlabel('number of optimizations/time')#(优化次数/代)
    plt.ylabel('learning rate')#学习率
    
    plt.figure()
    plt.plot(result[:,1])
    plt.xlabel('number of optimizations/time')
    plt.ylabel('number of training')#训练次数
    
    plt.figure()
    plt.plot(result[:,2])
    plt.xlabel('number of optimizations/time')
    plt.ylabel('Batchsize')
    
    plt.figure()
    plt.plot(result[:,3])
    plt.xlabel('number of optimizations/time')
    plt.ylabel('hiddens of LSTM1')
    
    plt.figure()
    plt.plot(result[:,4])
    plt.xlabel('number of optimizations/time')
    plt.ylabel('hiddens of LSTM2')

    plt.show()
HUATU(trace,process)


# In[7]:


tf.random.set_seed(2021) # 全局设置tf.set_random_seed()函数，使用之后后面设置的随机数都不需要设置seed，而可以跨会话生成相同的随机数。


lr = best[0]  # 学习率
num_epochs = int(best[1])  # 迭代次数
batch_size = int(best[2])  # batchsize
hidden1 = int(best[3])  # lstm1隐藏层节点数
hidden2 = int(best[4])  # lstm2隐藏层节点数

window, feature = X_train.shape[-2:]

inputs = Input(shape=(window, feature))  # 输入形状(window,input_size)
model=Conv1D(filters = 7, kernel_size = 2,strides=1,kernel_initializer=initializers.glorot_uniform(seed=2022))(inputs)#卷积层
model=MaxPooling1D(pool_size = 3,strides=2)(model)#池化层
model=LSTM(units=hidden1,return_sequences=True)(model)#LSTM1
model=LSTM(units=hidden2,return_sequences=True)(model)#LSTM2
#model=Dropout(0.3)(model)
attention=Dense(hidden2, activation='sigmoid', name='attention_vec')(model)#求解Attention权重
model=Multiply()([model, attention])#attention与LSTM对应数值相乘
model=Flatten()(model)  #flatten()：Flatten层用来将输入压平，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
outputs = Dense(3,kernel_initializer=initializers.glorot_uniform(seed=2022))(model)  # 全连接层
model = Model(inputs=inputs, outputs=outputs)  # 网络model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

model.summary()  # 展示模型结构
start = timer()
history = model.fit(X_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_data=(X_test, y_test), verbose=1)
end = timer()
print(end - start)
#


# In[8]:


model.save("Model/cnn_woa_lstm_attention_model.h5") # 该方法能够将整个模型进行保存：模型结构，能够重新实例化模型；模型权重；优化器的状态，在上次中断的地方继续训练；


# In[9]:


import json
# json.dump(history.history,open("woa_lstm_10_history.json","w")) # 将json格式的数据写进文件里

with open("cnn_woa_lstm_attention_history.json", "w") as file:  
    json.dump(history.history, file)


# In[10]:


plt.rcParams['font.family'] = ['sans-serif'] # font.family设置字体样式
plt.rcParams['font.sans-serif'] = ['SimHei'] #font.sans-serif设置字体
plt.rcParams['figure.dpi'] = 600  # 图像分辨率


# In[11]:


import json
history_load=json.load(open("cnn_woa_lstm_attention_history.json","r")) # json.load()：用来读取文件的，即，将文件打开然后就可以直接读取。
loss = history_load['loss'] # 训练集的每一轮的损失值
val_loss = history_load['val_loss'] # 测试集的损失值
epochs_range = range(len(loss))  # range()函数：它能返回一系列连续增加的整数
plt.plot(epochs_range, loss, label='Train Loss')  # 训练损失
plt.plot(epochs_range, val_loss, label='Test Loss')  # 测试损失
plt.legend(loc='upper right')  #设置图像的位置
plt.title('Train and Val Loss')
plt.legend() #显示标签
plt.show()


# In[12]:


def mape(y_true, y_pred):  # mape:平均绝对百分比误差
    y_t_index = y_true != 0
    y_t = y_true[y_t_index] # 测试集目标真实值
    y_p = y_pred[y_t_index] # 测试集目标预测值
    return np.mean(np.abs((y_p - y_t) / y_t))


# In[13]:


model_restored=tf.keras.models.load_model("Model/cnn_woa_lstm_attention_model.h5") # 重新实例化保存的模型，通过该方法返回的模型是已经编译过的模型
 # 当使用predict()方法进行预测时，返回值是数值，表示样本属于每一个类别的概率
 # scaler.inverse_transform()：将标准化后的数据转换为原始数据
test_pre=scaler.inverse_transform(model_restored(X_test)) # 预测数据反归一化
test_true=scaler.inverse_transform(y_test) # 真实数据反归一化

#print(model_restored(X_test))
#print(model_restored.predict(X_test))


# In[14]:


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

# test_pre_t4=test_pre[:,3]
# test_true_t4=test_true[:,3]
# print('测试集上死亡人数的MAE/RMSE/MAPE/R square')
# print(mean_absolute_error(test_true_t4, test_pre_t4))
# print(np.sqrt(mean_squared_error(test_true_t4, test_pre_t4)))
# print(mape(test_true_t4, test_pre_t4))
# print(r2_score(test_true_t4, test_pre_t4))


# In[15]:


plt.plot(test_pre_t1, label='确诊人数_预测')  # 训练损失
plt.plot(test_true_t1, label='确诊人数_真实')  # 测试损失
plt.legend(loc='best')
plt.title('确诊人数')
plt.legend()
plt.show()

plt.plot(test_pre_t2, label='治愈人数_预测')  # 训练损失
plt.plot(test_true_t2, label='治愈人数_真实')  # 测试损失
plt.legend(loc='best')
plt.title('治愈人数')
plt.legend()
plt.show()

plt.plot(test_pre_t3, label='死亡人数_预测')  # 训练损失
plt.plot(test_true_t3, label='死亡人数_真实')  # 测试损失
plt.legend(loc='best')
plt.title('死亡人数')
plt.legend()
plt.show()

