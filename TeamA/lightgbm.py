# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from datetime import datetime
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
import numpy as np
import pickle

# %%
def loadData(path):
    trainData = pd.read_csv(path + "/train_data.csv", index_col=0)
    testData = pd.read_csv(path + "/test_data.csv", index_col=0)

    trainData = trainData.drop('d.rainfall_detect', axis=1)
    testData = testData.drop('d.rainfall_detect', axis=1)

    missing = [i for i, v in enumerate(trainData.loc[:, 'd.wind_speed']) if v == -9999.0]
    for i in missing:
        trainData.loc[i, 'd.wind_speed'] = np.nan
    wind_speed_mean = trainData.loc[:, 'd.wind_speed'].mean()

    missing = [i for i, v in enumerate(testData.loc[:, 'd.wind_speed']) if v == -9999.0]
    for i in missing:
        testData.loc[i, 'd.wind_speed'] = np.nan
    trainData.iloc[:, 0] = trainData.iloc[:, 0].map(lambda x: int(x[11:13]))
    testData.iloc[:, 0] = testData.iloc[:, 0].map(lambda x: int(x[11:13]))

    trainData = trainData.fillna(wind_speed_mean)
    testData = testData.fillna(wind_speed_mean)
    X_data = trainData.iloc[:, :18].to_numpy()
    y_data = trainData.iloc[:, 18:].to_numpy()
    X_test = testData.iloc[:, :18].to_numpy()

    return X_data, y_data, X_test


# %%
class dataset:
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def setData(self, x, y):
        self.x = x
        self.y = y


# %%


def objective(args):
    clf, score = train(args)
    return -score


space = {
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss', 'rf']),
    'learning_rate': hp.uniform('learning_rate', 0.01, 5),
    'max_depth': hp.choice('max_depth', range(3, 20, 1)),
    'num_leaves': hp.choice('num_leaves', range(10, 300, 10)),
    'n_estimators': hp.choice('n_estimators', range(10, 300, 5)),
    'subsample_for_bin': hp.choice('subsample_for_bin', range(1000, 10000, 1000)),
    'min_split_gain': hp.uniform('min_split_gain', 0.1, 5),
    'min_child_weight': hp.choice('min_child_weight', [1e-3, 3e-3, 5e-3]),
    'min_child_samples': hp.choice('min_child_samples', range(20, 100, 10)),
    'subsample': hp.uniform('subsample', 0.01, 1),
    'subsample_freq': hp.choice('subsample_freq', range(0, 10, 1)),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0.1, 2),
    'reg_lambda': hp.uniform('reg_lambda', 0.1, 2),
}


# %%
def getHyper(trainDataset, validDataset):
    bestParms = []
    labelSize = y_train.shape[-1]
    for i in range(labelSize):
        print(f'Feature: {i}')
        trainDataset.setData(X_train, y_train[:, i])
        validDataset.setData(X_valid, y_valid[:, i])
        best = fmin(objective, space, algo=tpe.suggest, max_evals=50)
        bestParms.append(space_eval(space, best))
    return bestParms


def train(args):
    clf = LGBMClassifier(boosting_type='dart',
                         learning_rate=args['learning_rate'],
                         max_depth=args['max_depth'],
                         num_leaves=args['num_leaves'],
                         n_estimators=args['n_estimators'],
                         subsample_for_bin=args['subsample_for_bin'],
                         min_split_gain=args['min_split_gain'],
                         min_child_weight=args['min_child_weight'],
                         min_child_samples=args['min_child_samples'],
                         subsample=args['subsample'],
                         subsample_freq=args['subsample_freq'],
                         colsample_bytree=args['colsample_bytree'],
                         reg_alpha=args['reg_alpha'],
                         reg_lambda=args['reg_lambda'],
                         random_state=int(datetime.now().timestamp()))

    clf.fit(trainDataset.x, trainDataset.y)
    score = clf.score(validDataset.x, validDataset.y)
    return clf, score


# %%
X_data, y_data, X_test = loadData('IA')
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8,
                                                      random_state=int(datetime.now().timestamp()))
trainDataset = dataset(None, None)
validDataset = dataset(None, None)
# %%
bestParms = getHyper(trainDataset, validDataset)
# %%

modelList = []
validPrediction = []
testPrediction = []
for i, hyperParms in enumerate(bestParms):
    trainDataset.setData(X_train, y_train[:, i])
    validDataset.setData(X_valid, y_valid[:, i])
    clf, score = train(hyperParms)
    modelList.append(clf)
    validPrediction.append(clf.predict(X_valid))
    testPrediction.append(clf.predict(X_test))


with open('modelList.pickle', 'wb') as handle:
    pickle.dump(modelList, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('modelList.pickle', 'rb') as handle:
    modelList = pickle.load(handle)
validPrediction = np.array(validPrediction).T
testPrediction = np.array(testPrediction).T
# %%
f1 = f1_score(validPrediction, y_valid, average='micro', zero_division=True)

# %%
sample = pd.read_csv('IA/submission.csv', index_col=0)
# %%
submission = pd.DataFrame(testPrediction, columns=sample.columns.to_list())
# %%
submission.to_csv('IA/lightbgmPrediction.csv')
