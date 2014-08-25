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

X, y = prepare_data.load('train.csv')

clf1 = RandomForestClassifier(n_estimators=3000, max_depth=6,
                              min_samples_leaf=1,
                              max_features=1., min_samples_split=1)
clf2 = SVC(C=1., gamma=0.2)
clf3 = GradientBoostingClassifier(n_estimators=3000, max_depth=6,
                                  min_samples_leaf=1,
                                  max_features=1., min_samples_split=1)

predictions = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=i, test_size=0.2)
    # Scale
    scalerx = StandardScaler().fit(X_train)
    X_train_scaled = scalerx.transform(X_train)
    X_test_scaled = scalerx.transform(X_test)
    clf1.fit(X_train_scaled, y_train)
    clf2.fit(X_train_scaled, y_train)
    clf3.fit(X_train_scaled, y_train)
    y_predict1 = clf1.predict(X_test_scaled)
    y_predict2 = clf2.predict(X_test_scaled)
    y_predict3 = clf3.predict(X_test_scaled)
    y_predict = np.round(np.mean(np.asarray(zip(
        y_predict1, y_predict2, y_predict3)), axis=1)).astype(int)
    acc = accuracy_score(y_test, y_predict)
    print acc
    predictions.append(acc)
print 'All: ', np.asarray(predictions).mean()
