import pandas as pd
import numpy as np
import scipy as sp
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll

def estimate_missing(df):

    df.Age.fillna(df.Age.median(), inplace=True)

    # Set embarked to max passengers embarked
    max_embarked = df.groupby('Embarked').count()['PassengerId']
    df.Embarked.fillna(max_embarked[max_embarked == max_embarked.max()].index[0], inplace=True)

    df.Fare.fillna(df.Fare.median(), inplace=True)

    # Too much missing values, get rid of them
    df = df.drop(['Ticket', 'Cabin'], axis=1)

    return df


def generate_features(df):

    df['Sex_vect'], _ = pd.factorize(df['Sex'])

    return df


def main():
    df = pd.read_csv('data/train.csv')

    df = generate_features(df)
    df = estimate_missing(df)

    df.dropna(inplace=True)

    features = np.array(['Age', 'Fare', 'Sex_vect', 'Pclass', 'SibSp', 'Parch'])

    clf = RandomForestClassifier(n_estimators=80, criterion='entropy', max_depth=5)

    x_train = df[features]
    y_train = df['Survived']

    cv = cross_validation.KFold(len(df), n_folds=5, indices=False)

    results = []
    accs = []

    for traincv, testcv in cv:
        probas = clf.fit(x_train[traincv], y_train[traincv]).predict_proba(x_train[testcv])
        results.append(llfun(y_train[testcv], [x[1] for x in probas]))
        accs.append(clf.score(x_train, y_train))

    print "Results: " + str(np.array(results).mean())
    print('Accuracy: ' + str(np.array(accs).mean()))

    test_df = pd.read_csv('data/test.csv')

    test_df = generate_features(test_df)
    test_df = estimate_missing(test_df)

    test_df.fillna(0, inplace=True)

    clf.fit(x_train, y_train)

    print('Accuracy on all data: ' + str(clf.score(x_train, y_train)))

    # plot_importance(clf, df, features)

    test_df['Survived'] = clf.predict(test_df[features])

    test_df.to_csv('data/submission.csv', cols=['PassengerId', 'Survived'], index=False, float_format="%d")


def plot_importance(clf, train_df, features):
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    pl.subplot(1, 2, 2)
    pl.barh(pos, feature_importance[sorted_idx], align='center')
    pl.yticks(pos, train_df[features].columns[sorted_idx])
    pl.xlabel('Relative Importance')
    pl.title('Variable Importance')
    pl.show()


if __name__ == '__main__':
    main()