# 2016.9.2, use python to tackle this dataset

import pandas as pd
import numpy as np
def get_data(name,chunksize):
    reader_numeric = pd.read_csv(name+'_numeric.csv', chunksize=chunksize)
    reader_categorical = pd.read_csv(name+'_categorical.csv', chunksize=chunksize)
    reader_date = pd.read_csv(name+'_date.csv', chunksize=chunksize)
    reader = zip(reader_numeric, reader_categorical, reader_date)
    first = True
    for numeric, categorical, date in reader:
        categorical.drop('Id', axis=1, inplace=True)
        date.drop('Id', axis=1, inplace=True)
        data = pd.concat([numeric, categorical, date], axis=1)
        merge_data = data
        if first:
             merge = merge_data.copy()
             first = False
        else:
             merge = pd.concat([merge, merge_data])
    return merge

train = get_data('train',10000)
test = get_data('test',10000)

# Or read it individually
name='train'
train_num=pd.read_csv(name+'_numeric.csv')
#train_date=pd.read_csv(name+'_date.csv')
#train_cate=pd.read_csv(name+'_categorical.csv')
name='test'
test_num=pd.read_csv(name+'_numeric.csv')

# dimension
train_num.shape # (1183747, 970)

# system time
from datetime import datetime                                                                                                                            |
datetime.now().strftime('%Y-%m-%d %H:%M:%S') 

# install scikit-learn
pip install -U scikit-learn

# train dataset: deal with missing values by Imputer (mean)
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_num.iloc[:,1:969])
train_fill=imp.transform(train_num.iloc[:,1:969])
train_num=train_num.iloc[:,[0,969]] # release momery?

# try decision tree
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(train_fill,train_num.Response)

# save and load workspace for memory
import pickle
with open('dtc_python', 'wb') as f:                                                                                                                     |
    pickle.dump(dtc, f)
quit()
with open('dtc_python','rb') as f:                                                                                                                      |
...     dtc=pickle.load(f) 
# However, in this way, one has to remember the order of every object

# test dataset
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(test_num.iloc[:,1:969])
test_fill=imp.transform(test_num.iloc[:,1:969])
test_num=test_num.iloc[:,0:1] # release momery?

# predict by decision tree
test_dtc=dtc.predict(test_fill) # score: 0.11328

# output
f=open('mean_fill_dtc.csv','w')
f.write("Id,Response\n")
for id,res in zip(test_num.Id,test_dtc):
    tmp=f.write("%d,%d\n" %(id,res))
f.close()
# as a function
def out_predict(out_file,name,predict):
    f=open(out_file,'w')
    f.write("Id,Response\n")
    for id,res in zip(name,predict):
        tmp=f.write("%d,%d\n" %(id,res))
    f.close()


# 2016.9.13, try random forest
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=10)
rfc.fit(train_fill,train_num.Response)
# save the random forest
import pickle
pickle.dump(rfc, open('rfc_python', 'wb'))
quit()
rfc=pickle.load(open('rfc_python','rb')) 

# predict and output
test_rfc=rfc.predict(test_fill)
out_predict(out_file='mean_fill_rfc.csv',name=test_num.Id,predict=test_rfc) # score: 0.13326

# try Extremely Randomized Trees
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier(n_estimators=10)
etc.fit(train_fill,train_num.Response)
import pickle
pickle.dump(etc, open('etc_python', 'wb'))
quit()
etc=pickle.load(open('etc_python','rb')) 

# predict and output
test_etc=etc.predict(test_fill)
out_predict(out_file='mean_fill_etc.csv',name=test_num.Id,predict=test_etc) # score: 0.15185
# n_estmators=50, score: 0.15279

# 2016.9.14, try Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train_fill,train_num.Response) # out of memory
# use partial_fit
gnb.partial_fit(train_fill[0:200000,:],train_num.Response[0:200000],classes=[0,1])
gnb.partial_fit(train_fill[200000:400000,:],train_num.Response[200000:400000],classes=[0,1])
gnb.partial_fit(train_fill[400000:600000,:],train_num.Response[400000:600000],classes=[0,1])
gnb.partial_fit(train_fill[600000:800000,:],train_num.Response[600000:800000],classes=[0,1])
gnb.partial_fit(train_fill[800000:1000000,:],train_num.Response[800000:1000000],classes=[0,1])
gnb.partial_fit(train_fill[1000000:1183747,:],train_num.Response[1000000:1183747],classes=[0,1])
import pickle
pickle.dump(gnb, open('gnb_python', 'wb'))
quit()
gnb=pickle.load(open('gnb_python','rb')) 
# predict and output
test_gnb=gnb.predict(test_fill)
out_predict(out_file='mean_fill_gnb.csv',name=test_num.Id,predict=test_gnb) # score: 0.02537, I think it is heavily biased by filling the missing values. Or many features are not independent to each other.

# try support vector machine
from sklearn import svm
svmc=svm.SVC()
svmc.fit(train_fill,train_num.Response)
import pickle
pickle.dump(svmc, open('svmc_python', 'wb'))
quit()
svmc=pickle.load(open('svmc_python','rb')) 
# predict and output
test_svmc=svmc.predict(test_fill)
out_predict(out_file='mean_fill_svmc.csv',name=test_num.Id,predict=test_svmc) # score: 0, all predictions are 0...
