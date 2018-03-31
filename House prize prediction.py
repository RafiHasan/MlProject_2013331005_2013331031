
# coding: utf-8

# In[4]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

data = pd.read_csv("kshouses.csv")
data = data.drop(['date'],axis=1)
data.describe()


reg = LinearRegression()
labels = data['price']
train1 = data.drop(['price'],axis=1)
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.30,random_state =2)
reg.fit(x_train,y_train)
result=reg.score(x_test,y_test)
print( round(result*100,2))


rfr = RandomForestRegressor(n_estimators=8, max_depth=8, random_state=9, verbose=0)
rfr.fit(x_train, y_train)
acc=rfr.score(x_test,y_test)
print(round(acc*100,2))


reg = ElasticNet()
reg.fit(x_train,y_train)
result=reg.score(x_test,y_test)
print( round(result*100,2))


sgd = LassoLars()
sgd.fit(x_train,y_train)
result=sgd.score(x_test,y_test)
print( round(result*100,2))

reg = HuberRegressor()
reg.fit(x_train,y_train)
result=reg.score(x_test,y_test)
print( round(result*100,2))


reg = Ridge()
reg.fit(x_train,y_train)
result=reg.score(x_test,y_test)
print( round(result*100,2))


reg = Lasso(alpha=0.1)
reg.fit(x_train,y_train)
result=reg.score(x_test,y_test)
print( round(result*100,2))



reg = BayesianRidge()
reg.fit(x_train,y_train)
result=reg.score(x_test,y_test)
print( round(result*100,2))



reg = GradientBoostingRegressor()
reg.fit(x_train,y_train)
result=reg.score(x_test,y_test)
print( round(result*100,2))


RG = RidgeCV(alphas=(0.2, 2.0, 20.0), cv=None, fit_intercept=True, gcv_mode=None,
    normalize=False, scoring=None, store_cv_values=False)
RG.fit(x_train,y_train)
result=RG.score(x_test,y_test)
print( round(result*100,2))


LL = LassoLarsCV(copy_X=True, cv=None, eps=2.2204460492503131e-16,
      fit_intercept=True, max_iter=500, max_n_alphas=1000, n_jobs=1,
      normalize=True, positive=False, precompute='auto', verbose=False)
LL.fit(x_train,y_train)
result=LL.score(x_test,y_test)
print( round(result*100,2))


LSC = LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
LSC.fit(x_train,y_train)
result=LSC.score(x_test,y_test)
print( round(result*100,2))
