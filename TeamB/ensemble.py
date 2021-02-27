from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier
from sklearn import  ensemble, preprocessing, metrics
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

modelsName = ["SVC","DecisionTree","AdaBoost","Bagging","RandomForest","ExtraTrees",,"GradientBoosting","MultipleLayerPerceptron","XGBClassifier","LogisticRegression","KNeighbors","LinearDiscriminantAnalysis"]

kfold = StratifiedKFold(n_splits=12)
random_state = 31
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(random_state=random_state)
classifiers.append(BaggingClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(XGBClassifier(random_state=random_state))
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, np.array(featureX), y = np.array(labelX09), scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    print(cv_means[len(cv_means)-1])

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":modelsName})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")






#SVM
model09 = svm.SVC(decision_function_shape='ovo')
model09.fit(featureX,labelX09)
predictions09 = model09.predict(featureY)
predictions_all.append(predictions09)
accuracy = metrics.accuracy_score(labelY09, predictions09)
print(accuracy)
print(metrics.precision_recall_fscore_support(labelY09, predictions09,average='micro'))


#ExtraTreesClassifier
model09 = ExtraTreesClassifier(criterion='entropy',n_estimators=350,n_jobs=-1,max_depth=None,random_state=31)
model09.fit(featureX,labelX09)
predictions09 = model09.predict(featureY)
predictions_all.append(predictions09)
accuracy = metrics.accuracy_score(labelY09, predictions09)
print(accuracy)
print(metrics.precision_recall_fscore_support(labelY09, predictions09,average='micro'))

#AdaBoostClassifier
model09 = AdaBoostClassifier(n_estimators=350,random_state=31)
model09.fit(featureX,labelX09)
predictions09 = model09.predict(featureY)
predictions_all.append(predictions09)
accuracy = metrics.accuracy_score(labelY09, predictions09)
print(accuracy)
print(metrics.precision_recall_fscore_support(labelY09, predictions09,average='micro'))

#GradientBoostingClassifier
model09 = GradientBoostingClassifier(n_estimators=350, random_state=31)
model09.fit(featureX,labelX09)
predictions09 = model09.predict(featureY)
predictions_all.append(predictions09)
accuracy = metrics.accuracy_score(labelY09, predictions09)
print(accuracy)
print(metrics.precision_recall_fscore_support(labelY09, predictions09,average='micro'))

#BaggingClassifier
model09 = BaggingClassifier(n_estimators=350, random_state=31)
model09.fit(featureX,labelX09)
predictions09 = model09.predict(featureY)
predictions_all.append(predictions09)
accuracy = metrics.accuracy_score(labelY09, predictions09)
print(accuracy)
print(metrics.precision_recall_fscore_support(labelY09, predictions09,average='micro'))

#XGBClassifier
model09 = XGBClassifier(n_estimators=30,random_state=31)
model09.fit(np.array(featureX),np.array(labelX09))
predictions09 = model09.predict(np.array(featureY))
predictions_all.append(predictions09)
accuracy = metrics.accuracy_score(labelY09, predictions09)
print(accuracy)
print(metrics.precision_recall_fscore_support(labelY09, predictions09,average='micro'))

predictions_vote = []

for r in range(len(predictions09)):
    vote=0
    for i in range(5):
        vote += predictions_all[i][r]
    if vote >= 3:
        predictions_vote.append(1)
    else:
        predictions_vote.append(0)

accuracy = metrics.accuracy_score(labelY09, predictions_vote)
print(accuracy)
print(metrics.precision_recall_fscore_support(labelY09, predictions_vote,average='micro'))

predictions09 = np.array([[i] for i in predictions_vote])

#for idx,d in enumerate(predictions):
predictions=np.concatenate ((predictions,predictions09),axis=1)

for idx,l in enumerate(labelY):
    labelY[idx].append(labelY09[idx])