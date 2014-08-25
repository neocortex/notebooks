import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier,\
    RandomForestClassifier
from sklearn.lda import LDA
from sklearn.grid_search import GridSearchCV


def load(fname='train.csv'):
    # Load data
    raw_data = pd.read_csv(fname)
    # Remove columns
    ids = raw_data.PassengerId.values
    raw_data = raw_data.drop('PassengerId', axis=1)
    raw_data = raw_data.drop('Ticket', axis=1)
    raw_data = raw_data.drop('Name', axis=1)
    raw_data = raw_data.drop('Cabin', axis=1)

    # Fill NaNs with medians
    medians = raw_data.mean(skipna=True)
    data = raw_data.fillna(medians)

    # Fill NaNs of embark with most frequent value
    data.Embarked = data.Embarked.fillna(data.Embarked.value_counts()[0])

    # Replace categorical variables with dummies
    new_sex_column = pd.get_dummies(data.Sex, prefix='Sex')
    data = data.drop('Sex', axis=1)
    data = pd.concat((data, new_sex_column), axis=1)
    #data = data.rename(columns={'Sex_male': 'Sex'})
    #data = data.drop('Sex_female', axis=1)
    new_embarked_column = pd.get_dummies(data.Embarked, prefix='Embarked')
    if hasattr(new_embarked_column, 'Embarked_644'):
        new_embarked_column = new_embarked_column.drop('Embarked_644', axis=1)
    data = data.drop('Embarked', axis=1)
    data = pd.concat((data, new_embarked_column), axis=1)
    #data = data.drop('SibSp', axis=1)
    #data = data.drop('Parch', axis=1)
    # Extract targets
    if hasattr(data, 'Survived'):
        y = data.Survived.values
        data = data.drop('Survived', axis=1)
        return data.values, y
    else:
        return data.values, ids


def fit():
    X, y = load('train.csv')
    # Scale
    #scalerx = StandardScaler().fit(X)
    #X = scalerx.transform(X)

    # Classifier
    # clf = GradientBoostingClassifier(n_estimators=3000)
    clf = RandomForestClassifier(n_estimators=3000)
    # clf = SVC()
    # clf = LDA()
    # clf = LogisticRegression(C=10.)

    # Gridsearch parameters
    param_grid_forest = {'min_samples_leaf': [1, 2, 3, 5],
                         'max_depth': [5, 6, 7],
                         'max_features': [1., .9, .8, .5],
                         'min_samples_split': [5, 6, 7, 8]}
    param_grid_boosting = {'learning_rate': [0.03, 0.02, 0.01],
                           'max_depth': [3, 4, 5, 6],
                           'max_features': [.7, .6, .5],
                           'min_samples_split': [6, 8, 10],
                           'subsample': [1., .9, 0.8, 0.7]}
    param_grid_svc = {'C': [0.5, 1, 1.5, 2, 3, 5, 10],
                      'gamma': [0.01, 0.1, 0.2, 0.25, 0.3, .5],
                      'kernel': ['rbf', 'sigmoid']}

    # Gridsearch
    gs = GridSearchCV(clf, param_grid=param_grid_forest, cv=5,
                      verbose=3, n_jobs=2)
    gs.fit(X, y, cv=5, verbose=5)
    scores = cross_val_score(gs.best_estimator_, X, y, cv=10)
    print scores.min(), scores.mean(), scores.max()
    return gs
