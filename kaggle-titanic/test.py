import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.lda import LDA
from sklearn.metrics import accuracy_score
import prepare_data

X_train, y_train = prepare_data.load('train.csv')

clf1 = RandomForestClassifier(n_estimators=3000, max_depth=5, min_samples_leaf=1,
                              max_features=1., min_samples_split=1)
clf2 = SVC(C=1., gamma=0.2)
clf3 = GradientBoostingClassifier(n_estimators=3000, max_depth=6,
                                  min_samples_leaf=1,
                                  max_features=1., min_samples_split=1)

# Scale
#scalerx = StandardScaler().fit(X_train)
#X_train = scalerx.transform(X_train)
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
# Predict
X_test, ids = prepare_data.load('test.csv')
#X_test = scalerx.transform(X_test)
y_predict1 = clf1.predict(X_test)
y_predict2 = clf2.predict(X_test)
y_predict3 = clf3.predict(X_test)
y_predict = np.round(np.mean(np.asarray(zip(
    y_predict1, y_predict2, y_predict3)), axis=1)).astype(int)

import csv
with open('predictions.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    header = ['PassengerId', 'Survived'],
    a.writerows(header)
    a.writerows(zip(ids, y_predict))
