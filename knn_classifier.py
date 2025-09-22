from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 25)
model.fit(X_train,y_train)
dt_knn=model.predict(X_test)
kfold = KFold(n_splits=10, random_state=None) # k=10, split the data into 10 equal parts=
result_knn=cross_val_score(model,scaled_features,target_feature,cv=kfold,scoring='accuracy')
print('The overall score for K Nearest Neighbors Classifier is:',round(result_knn.mean()*100,2))
y_pred = cross_val_predict(model,scaled_features,target_feature,cv=10)
sns.heatmap(confusion_matrix(dt_knn,y_test),annot=True,fmt=".1f",cmap='summer')
plt.title('KNN Confusion_matrix')
#KNN fold accuracy visualizer
_result_knn=[r*100 for r in result_knn]
plt.plot(_result_knn)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('K-NN fold accuracy visualizer')
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score,
recall_score, f1_score
print('Micro Precision: {:.4f}'.format(precision_score(y_test, dt_knn, average='micro')))
print('Micro Recall: {:.4f}'.format(recall_score(y_test, dt_knn, average='micro')))
print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, dt_knn, average='micro')))
print('Macro Precision: {:.4f}'.format(precision_score(y_test, dt_knn, average='macro')))
print('Macro Recall: {:.4f}'.format(recall_score(y_test, dt_knn, average='macro')))
print('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, dt_knn, average='macro')))
print('Weighted Precision: {:.4f}'.format(precision_score(y_test, dt_knn, average='weighted')))
print('Weighted Recall: {:.4f}'.format(recall_score(y_test, dt_knn, average='weighted')))
print('Weighted F1-score: {:.4f}'.format(f1_score(y_test, dt_knn, average='weighted')))
print('\n--------------- K-Nearest Neighbour Classification Report ---------------\n')
print(classification_report(y_test, dt_knn))
#print('---------------------- K-NN ----------------------') # unnecessary fancy styling
