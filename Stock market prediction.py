import numpy as np

import glob
import csv
import operator
from sklearn.linear_model import LinearRegression
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoCV

path =r'2012' # use your path
allFiles = glob.glob(path + "/*.csv")
mlist=[]
dt=[]
cp=[]
dat=0
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat
            mlist.append(o)
path =r'2013' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat
            mlist.append(o)
path =r'2014' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat            
            mlist.append(o)
path =r'2015' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
      #  print(dat)
        for o in reader:
            o[1]=dat 
            mlist.append(o)
sorlist=sorted(mlist, key=operator.itemgetter(0,1), reverse=False)
na=np.array(sorlist)


trainx=[]
trainy=[]

x=0
y=na[0][0]
list1=[]
list2=[]
for i in na:    
    if(y!=i[0]):
       # print(list2)
        y=i[0]
        trainx.append(list1)
        trainy.append(list2)
        list1=[]
        list2=[]
    try: 
        kx=[]
        kx.append(float(i[1]))
        kx.append(float(i[5]))
        kx.append(float(i[6]))
        list1.append(kx)
        list2.append(float(i[5]))
    except:
        print("")


for k in range(14,15):
    print("..........................",k,".......................")
    reg = LinearRegression()
    reg.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=reg.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+((result[i-1]-trainy[k][i])/trainy[k][i])**2 
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    rfr = RandomForestRegressor(n_estimators=8, max_depth=8, random_state=9, verbose=0)
    rfr.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=rfr.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    reg = ElasticNet()
    reg.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=reg.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    sgd = LassoLars()
    sgd.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=sgd.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    reg = Ridge()
    reg.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=reg.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    reg = Lasso(alpha=0.1)
    reg.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=reg.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    
    reg = BayesianRidge()
    reg.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=reg.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    
    reg = GradientBoostingRegressor()
    reg.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=reg.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    RG = RidgeCV(alphas=(0.2, 2.0, 20.0), cv=None, fit_intercept=True, gcv_mode=None,
        normalize=False, scoring=None, store_cv_values=False)
    RG.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=RG.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    LL = LassoLarsCV(copy_X=True, cv=None, eps=2.2204460492503131e-16,
          fit_intercept=True, max_iter=500, max_n_alphas=1000, n_jobs=1,
          normalize=True, positive=False, precompute='auto', verbose=False)
    LL.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=LL.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)
    
    
    LSC = LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
        max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
        precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
        verbose=False)
    LSC.fit(trainx[k][0:int(len(trainx[k])/2)],trainy[k][1:int(len(trainy[k])/2)+1])
    result=LSC.predict(trainx[k])
    acc = 0
    for i in range(int(len(trainx[k])/2),len(result)):
        acc=acc+(result[i-1]/trainy[k][i]-trainy[k][i]/trainy[k][i])**2    
    acc=acc/int(len(result)/2)
    acc=acc**(1/2.0)
    print(1-acc)