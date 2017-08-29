#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.grid_search import GridSearchCV
import scipy as sp

def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll

train = pd.read_csv('pre/train.csv')
ad = pd.read_csv('pre/ad.csv')
user = pd.read_csv('pre/user.csv')
position = pd.read_csv('pre/position.csv')
test = pd.read_csv('pre/test.csv')

train = pd.merge(train,ad,on='creativeID',how='inner')
train = pd.merge(train,user,on='userID',how='inner')
train = pd.merge(train,position,on='positionID',how='inner')
train = train.drop(['clickTime','conversionTime','creativeID','userID','positionID','adID','camgaignID','appID','hometown','residence'],axis = 1)
test = pd.merge(test,ad,on='creativeID',how='inner')
test = pd.merge(test,user,on='userID',how='inner')
test = pd.merge(test,position,on='positionID',how='inner')
test = test.drop(['clickTime','creativeID','userID','positionID','adID','camgaignID','appID','hometown','residence'],axis = 1)


train_pos = train[train['label']==1]
train_neg = train[train['label']==0]
print train_pos.size
print train_neg.size

x = train.values[:,1:]
y = train.values[:,0]

X_test = test.values[:,2:]
instance_id_test = test.values[:,0]

'''
enc = preprocessing.OneHotEncoder()
enc.fit(x)
X = enc.transform(x).toarray()
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.9)
data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 2}

bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
y_hat = bst.predict(data_test)
print y_hat
'''
#线性分类器
class_weight={0:0.5, 1:0.5}
linreg = LogisticRegression(penalty='l1',class_weight= class_weight)
model = linreg.fit(x_train, y_train)

print model

y_hat = linreg.predict_proba(np.array(x_test))[:,1]
print y_hat
print y_test
score = logloss(y_test,y_hat)
print score

# X_test = enc.transform(X_test).toarray()


#L2L1正则+交叉验证
# model = Lasso()
model = Ridge()

alpha_can = np.logspace(-3, 2, 10)
lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
lasso_model.fit(x, y)
print '验证参数：\n', lasso_model.best_params_

y_hat = lasso_model.predict(np.array(x_test))
score = logloss(y_test,y_hat)
print score

'''

# df = pd.DataFrame({"instanceID": instance_id_test, "proba": proba_test})
# df.to_csv("lr_baseline.csv", columns=["instanceID", "proba"], index=False,)



