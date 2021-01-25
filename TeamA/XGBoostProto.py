#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from datetime import datetime
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
#%%
trainData = pd.read_csv("IA/train_data.csv", index_col=0)
testData = pd.read_csv("IA/test_data.csv", index_col=0)

#%%
X_data = trainData.iloc[:, 1:19].to_numpy()
y_data = trainData.iloc[:, 19:].to_numpy()

X_test = testData.iloc[:, 1:19].to_numpy()

#%%


def objective(args):
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8, random_state=int(datetime.now().timestamp()))
    clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, use_label_encoder=False,
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
    return -clf.score(X_valid, y_valid)


space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 2),
        'n_estimators': hp.choice('n_estimators', range(1, 101, 5)),
        'max_depth': hp.choice('max_depth', range(3, 20, 1)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'lambda': hp.uniform('lambda', 0.1, 2),
        'subsample': hp.uniform('subsample', 0.1, 1),
        'gamma': hp.uniform('gamma', 0.1, 0.5)
}

#%%

best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
print(space_eval(space, best))


#%%
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8, random_state=int(datetime.now().timestamp()))

args = space_eval(space, best)
clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, use_label_encoder=False,
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
clf.score(X_valid, y_valid)
#%%
prediction = clf.predict(X_test)

#%%
sample = pd.read_csv('IA/submission.csv', index_col=0)
#%%
submission = pd.DataFrame(prediction, columns=sample.columns.to_list())
#%%
submission.to_csv('IA/prototype.csv')

