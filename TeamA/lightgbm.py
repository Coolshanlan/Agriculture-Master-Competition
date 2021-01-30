#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from datetime import datetime
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
import numpy as np
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
    clf_multiclass = MultiOutputClassifier(LGBMClassifier(learning_rate=args['learning_rate'],
                                                        max_depth=args['max_depth'],
                                                        num_leaves=args['num_leaves'],
                                                        ))

    clf_multiclass.fit(X_train, y_train)
    prediction = clf_multiclass.predict(X_valid)
    f1 = f1_score(prediction, y_valid, average='micro', zero_division=True)
    return -f1


space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 5),
        'max_depth': hp.choice('max_depth', range(3, 20, 1)),
        'num_leaves': hp.choice('num_leaves', range(10, 300, 10))
}

#%%
X_data, y_data, X_test = loadData()
#%%
best = fmin(objective, space, algo=tpe.suggest, max_evals=50)

print(best)
print(space_eval(space, best))

#%%

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8, random_state=int(datetime.now().timestamp()))


args = space_eval(space, best)

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8, random_state=int(datetime.now().timestamp()))
clf_multiclass = MultiOutputClassifier(LGBMClassifier(learning_rate=args['learning_rate'],
                                                    max_depth=args['max_depth'],
                                                    num_leaves=args['num_leaves'],
                                                    ))

clf_multiclass.fit(X_train, y_train)
prediction = clf_multiclass.predict(X_valid)
f1 = f1_score(prediction, y_valid, average='micro', zero_division=True)

#%%
sample = pd.read_csv('IA/submission.csv', index_col=0)
#%%
submission = pd.DataFrame(prediction, columns=sample.columns.to_list())
#%%
submission.to_csv('IA/lightbgmPrediction.csv')
