import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

data = pd.read_excel("data_for_ML_final.xlsx") # 读取数据
data=data.drop(columns=['Sample'])


target = 'Shannon'
features = list(filter(lambda x:x not in [target], data.columns))
X = data[features]
y = np.log(data[target])
y = data[target]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X.loc[:, features] = scaler.transform(X)

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
import xgboost as xgb
# 区分自变量和因变量
target = 'Shannon' # 因变量字段名称
features = list(filter(lambda x:x not in [target], data.columns)) # 自变量字段
X = data[features]
# y = np.log(data[target]) # 因变量对数化处理
y = data[target]
from sklearn.model_selection import train_test_split 
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=111)

t = xgb.XGBRegressor(random_state=111)
t.fit(train_x, train_y)
y = t.predict(X)

perm_importance = t.feature_importances_
perm_importance
name=X.columns.values
name
perm_importance=list(perm_importance)
name=list(name)
a1=0
a2=''
for i in range(len(perm_importance)-1):
    for j in range(i,len(perm_importance)):
        if perm_importance[j]<perm_importance[i]:
            a1=perm_importance[j]
            perm_importance[j]=perm_importance[i]
            perm_importance[i]=a1
            a2=name[j]
            name[j]=name[i]
            name[i]=a2

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 6),dpi=150)
plt.subplot(1, 1, 1)
N = 6
width = 0.7


sns.barplot(x=name, y=perm_importance, err_kws={'linewidth': 0.1})
plt.xticks(rotation = 90,fontsize=6)

for x, y1 in enumerate(perm_importance): 
    plt.text(x, y1+ 0.001, '%g' % round(y1, 4), ha='center',fontsize=5,rotation = 0)  
    
plt.yticks(fontsize=6)
plt.xlabel('Feature name',fontsize=9)
plt.ylabel('Score',fontsize=9)

def feature_choose_reg(X, y, top_k=9):
    """
    Filter the first k most important feature names
    """
    import xgboost as xgb
    DT = xgb.XGBRegressor(random_state=111)
    DT.fit(X, y)
    imp_temp = pd.DataFrame({'feature':DT.feature_names_in_,'imp_score':DT.feature_importances_})
    imp_temp.sort_values(by='imp_score',ascending=False,inplace=True)
    feature_all = imp_temp['feature'].to_list()
    feature_choosed = feature_all[:top_k]

    return feature_choosed

feature_choosed = feature_choose_reg(X,y,top_k=9)

from sklearn.model_selection import train_test_split 
X = X[feature_choosed]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=111) #25% of the data is divided into test sets that are used to evaluate model generalization performance


from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

best_model = xgb.XGBRegressor(random_state=0).set_params(random_state=111,max_depth=3,n_estimators=19,learning_rate=0.1)

best_model.fit(train_x, train_y)

train_pred = best_model.predict(train_x)
test_pred =best_model.predict(test_x)

mse_train = np.sqrt(mean_squared_error(y_true=train_y,y_pred=train_pred))
r2_train = r2_score(y_true=train_y,y_pred=train_pred)
mae_train=mean_absolute_error(y_true=train_y,y_pred=train_pred) 

mse_test = np.sqrt(mean_squared_error(y_true=test_y,y_pred=test_pred))
r2_test = r2_score(y_true=test_y,y_pred=test_pred)
mae_test=mean_absolute_error(y_true=test_y,y_pred=test_pred) 

pd.DataFrame({'RMSE':[mse_train,mse_test],'R2':[r2_train, r2_test],'MAE':[mae_train,mae_test]},index=['train','test'])

import joblib
def classification(data,name):
    a=list(data[name].unique())
    b=[]
    c=list(data[name])
    for i in c:
        judge=False
        for j in range(len(a)):
            if a[j]==i:
                b.append(j)
                judge=True
        if judge== False:
            b.append(len(a))
  
    data=data.drop(columns=[name])
    data.insert(loc=0, column=name, value=b)
    return data
joblib.dump(best_model, 'xgboost_best_model.pkl')
















