import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
hazel_df = pd.read_csv("file path.csv")
hazel_df.head()
#Feature selection
all_features = hazel_df.drop("CLASS",axis=1)
target_feature = hazel_df["CLASS"]
all_features.head()
#Dataset preprocessing
from sklearn import preprocessing
x = all_features.values.astype(float) #returns a numpy array of type float
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scaled_features = pd.DataFrame(x_scaled)
scaled_features.head()
#Decision tree
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
import seaborn as sns
X_train,X_test,y_train,y_test =
train_test_split(scaled_features,target_feature,test_size=0.25,random_state=40)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier(criterion='gini',
 min_samples_split=10,min_samples_leaf=1,
 max_features='auto')
model.fit(X_train,y_train)
dt_pred=model.predict(X_test)
kfold = KFold(n_splits=10, random_state=None) # k=10, split the data into 10 equal parts
result_tree=cross_val_score(model,scaled_features,target_feature,cv=10,scoring='accuracy')
print('The overall score for Decision Tree classifier is:',round(result_tree.mean()*100,2))
y_pred = cross_val_predict(model,scaled_features,target_feature,cv=10)
sns.heatmap(confusion_matrix(dt_pred,y_test),annot=True,fmt=".1f",cmap='summer')
plt.title('Decision Tree Confusion_matrix')
#DT fold accuracy visualizer
_result_tree=[r*100 for r in result_tree]
plt.plot(_result_tree)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('DT fold accuracy visualizer')
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score,
recall_score, f1_score
print('Micro Precision: {:.4f}'.format(precision_score(y_test, dt_pred, average='micro')))
print('Micro Recall: {:.4f}'.format(recall_score(y_test, dt_pred, average='micro')))
print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, dt_pred, average='micro')))
print('Macro Precision: {:.4f}'.format(precision_score(y_test, dt_pred, average='macro')))
print('Macro Recall: {:.4f}'.format(recall_score(y_test, dt_pred, average='macro')))
print('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, dt_pred, average='macro')))
print('Weighted Precision: {:.4f}'.format(precision_score(y_test, dt_pred, average='weighted')))
print('Weighted Recall: {:.4f}'.format(recall_score(y_test, dt_pred, average='weighted')))
print('Weighted F1-score: {:.4f}'.format(f1_score(y_test, dt_pred, average='weighted')))
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
print('\n--------------- Decision Tree Classification Report ---------------\n')
print(classification_report(y_test, dt_pred))
#print('---------------------- Decison Tree ----------------------') # unnecessary fancy styling
