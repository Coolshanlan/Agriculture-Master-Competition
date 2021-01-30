#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from datetime import datetime
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
import numpy as np
from sklearn.metrics import f1_score


#%%
def loadData():
    trainData = pd.read_csv("IA/train_data.csv", index_col=0)
    testData = pd.read_csv("IA/test_data.csv", index_col=0)

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

#%%


def objective(args):
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8, random_state=int(datetime.now().timestamp()))
    clf = MultiOutputClassifier(XGBClassifier(n_jobs=-1, use_label_encoder=False,
                                            n_estimators=args['n_estimators'],
                                            max_depth=args['max_depth'],
                                            learning_rate=args['learning_rate'],
                                            colsample_bytree=args['colsample_bytree'],
                                            colsample_bylevel=args['colsample_bylevel'],
                                            reg_lambda=args['lambda'],
                                            subsample=args['subsample'],
                                            gamma=args['gamma'],
                                            min_child_weight=args['min_child_weight'],
                                            tree_method='gpu_hist'))
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_valid)
    f1 = f1_score(prediction, y_valid, average='micro', zero_division=True)
    return -f1


space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 5),
        'n_estimators': hp.choice('n_estimators', range(1, 101, 5)),
        'max_depth': hp.choice('max_depth', range(3, 20, 1)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'lambda': hp.uniform('lambda', 0.1, 2),
        'subsample': hp.uniform('subsample', 0.1, 1),
        'gamma': hp.uniform('gamma', 0.1, 2)
}

#%%
X_data, y_data, X_test = loadData()
best = fmin(objective, space, algo=tpe.suggest, max_evals=50)

print(best)
print(space_eval(space, best))

#%%
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8, random_state=int(datetime.now().timestamp()))


args = space_eval(space, best)
clf = MultiOutputClassifier(XGBClassifier(n_jobs=-1, use_label_encoder=False,
                                            n_estimators=args['n_estimators'],
                                            max_depth=args['max_depth'],
                                            learning_rate=args['learning_rate'],
                                            colsample_bytree=args['colsample_bytree'],
                                            colsample_bylevel=args['colsample_bylevel'],
                                            reg_lambda=args['lambda'],
                                            subsample=args['subsample'],
                                            gamma=args['gamma'],
                                            min_child_weight=args['min_child_weight'],
                                            ))
clf.fit(X_train, y_train)
#%%
prediction = clf.predict(X_valid)
f1 = f1_score(prediction, y_valid, average='micro', zero_division=True)

#%%
sample = pd.read_csv('IA/submission.csv', index_col=0)
#%%
submission = pd.DataFrame(prediction, columns=sample.columns.to_list())
#%%
submission.to_csv('IA/XGBoostPrediction.csv')
