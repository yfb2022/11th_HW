# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다. #
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
X = wine_df.drop(columns='target')
y = wine_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = DecisionTreeClassifier(random_state=42)
gridsearch = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)

importances = gridsearch.best_estimator_.feature_importances_
indices = importances.argsort()[::-1]
features = X.columns
plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
sns.barplot(x= importances[indices], y= features[indices], palette='viridis')
plt.ylabel('Feature')
plt.show()

####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9, 15],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200, 300]
}

clf = XGBClassifier(random_state=42)
gridsearch = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)
