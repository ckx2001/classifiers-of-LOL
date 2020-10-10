import numpy
import sklearn
import pickle as pkl
import pandas
from scipy.stats import randint as randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

                       ###数据导入###
f = open("D://Users//u303//Desktop//new_data.csv")
df = pandas.read_csv(f)
df2=df.drop_duplicates()
dataset = df2.values



                      ###数据预处理###

#------重复值删除-------

f = open("D://Users//u303//Desktop//new_data.csv")
df = pandas.read_csv(f)
d=0
for i in df.duplicated():
    if i !=False:
        d+=1
print("The number of original duplicated datas is",d)
print("The rows number of original datas is",len(df))

df2=df.drop_duplicates()

d=0
for i in df2.duplicated():
    if i !=False:
        d+=1
print("The number of new duplicated datas is",d)
print("The rows number of new datas is",len(df2))

#缺失值检测
df2.isnull().any(axis=0)



       ###x_train y_train  x_test y_test的生成  are as follows:###
                 ###进行标准化的数据###

#MAXMIN标准化
df3 = df2.astype('float32')
tool = MinMaxScaler(feature_range=(0, 1))
df4 = tool.fit_transform(df3)

#取30000个训练、测试集
dataset = df4
x = []
for i in range(30000):
    x_temp = dataset[i][5:21]
    x.append(x_temp)
x = numpy.array(x)

y = []
for i in range(30000):
    y_temp = dataset[i][4]
    y.append(y_temp)
y = numpy.array(y)

#训练集、测试集分割
[x_train, x_test, y_train, y_test] = train_test_split(x, y, stratify=y,random_state=1)

print(y_train,len(y_train))
print(x_train,len(x_train))


###x_train y_train  x_test y_test的生成 above!!!###






'''

                  ###decision_tree###
from sklearn.tree import DecisionTreeClassifier

#调试过程
tt = []
aa = []
for i in range(10):
    clf_DT = DecisionTreeClassifier()

    time_start=time.time()
    clf_DT.fit(x_train,y_train)
    time_end=time.time()

    t = time_end - time_start
    print("time:",t)

    accuracy_DT = clf_DT.score(x_test, y_test)
    paras_DT_temp = clf_DT.get_params()
    print("accuracy:",accuracy_DT)
    tt.append(t)
    aa.append(accuracy_DT)

for k in range(10):
    print("the time of",k+1,"th training is",tt[k])
    print("The accuracy of ",k+1,"th training is ",aa[k],"\n")


tt_ = (tt[1]+tt[2]+tt[3]+tt[4]+tt[5]+tt[6]+tt[7]+tt[8]+tt[9]+tt[0])/10
aa_ = (aa[1]+aa[2]+aa[3]+aa[4]+aa[5]+aa[6]+aa[7]+aa[8]+aa[9]+aa[0])/10
print("the time of training averagely is ",tt_,"\n","the accuracy of training averagely is ",aa_)


print(paras_DT_temp)
'''

'''
               ###MLP###

from sklearn.neural_network import MLPClassifier
#调试
clf_MLP = MLPClassifier(max_iter=3000).fit(x_train,y_train)

paras_MLP = clf_MLP.get_params()

accuracy_MLP = clf_MLP.score(x_test, y_test)


#结果
print(paras_MLP)
print(accuracy_MLP)
'''


'''
                  ###两层的ANN###
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=6)
        self.output = nn.Linear(in_features=6, out_features=2)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.output(x)
        x = F.softmax(x)
        return x


model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


  ##10次ANN的训练，得到训练的平均准确率和平均时间##
tt = []
aa = []
for j in range(10):

    t1 = time.time()
    epochs = 100
    loss_arr = []
    for i in range(epochs):
        y_hat = model.forward(x_train)
        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t2=time.time()

    t=t2-t1
    tt.append(t)

    predict_out = model(x_test)
    _,predict_y = torch.max(predict_out, 1)

    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, predict_y)
    aa.append(score)

for k in range(10):
    print("the time of",k+1,"th training is",tt[k])
    print("The accuracy of ",k+1,"th training is ",aa[k],"\n")


tt_ = (tt[1]+tt[2]+tt[3]+tt[4]+tt[5]+tt[6]+tt[7]+tt[8]+tt[9]+tt[0])/10
aa_ = (aa[1]+aa[2]+aa[3]+aa[4]+aa[5]+aa[6]+aa[7]+aa[8]+aa[9]+aa[0])/10
print("the time of training averagely is ",tt_,"\n","the accuracy of training averagely is ",aa_)

'''
                ###ANN结束###





                   ###knn###
from sklearn.neighbors import KNeighborsClassifier

tt = []
aa = []

for i in range(10):
    t1 = time.time()
    
    clf_knn = KNeighborsClassifier()
    clf_knn.fit(x_train,y_train)
    t2=time.time()
    accuracy_knn = clf_knn.score(x_test, y_test)


    t=t2-t1
    tt.append(t)
    aa.append(accuracy_knn)


for k in range(10):
    print("the time of",k+1,"th training is",tt[k])
    print("The accuracy of ",k+1,"th training is ",aa[k],"\n")
    

tt_ = (tt[1]+tt[2]+tt[3]+tt[4]+tt[5]+tt[6]+tt[7]+tt[8]+tt[9]+tt[0])/10
aa_ = (aa[1]+aa[2]+aa[3]+aa[4]+aa[5]+aa[6]+aa[7]+aa[8]+aa[9]+aa[0])/10
print("the time of training averagely is ",tt_,"\n","the accuracy of training averagely is ",aa_)

paras_knn =clf_knn.get_params()
print(paras_knn)












