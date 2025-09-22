from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_train,y_train)
gnb_pred=model.predict(X_test)
kfold = KFold(n_splits=10, random_state=None) # k=10, split the data into 10 equal parts
result_gnb=cross_val_score(model,scaled_features,target_feature,cv=10,scoring='accuracy')
print('The overall score for Gaussian Naive Bayes classifier is:',round(result_gnb.mean()*100,2))
y_pred = cross_val_predict(model,scaled_features,target_feature,cv=10)
sns.heatmap(confusion_matrix(gnb_pred,y_test),annot=True,fmt=".1f",cmap='summer')
plt.title('Naive Bayes Confusion_matrix')
#Naive bayes fold accuracy visualizer
_result_gnb=[r*100 for r in result_gnb]
plt.plot(_result_gnb)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.title('Naive bayes fold accuracy visualizer')
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score,
recall_score, f1_score
print('Micro Precision: {:.4f}'.format(precision_score(y_test, gnb_pred, average='micro')))
print('Micro Recall: {:.4f}'.format(recall_score(y_test, gnb_pred, average='micro')))
print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, gnb_pred, average='micro')))
print('Macro Precision: {:.4f}'.format(precision_score(y_test, gnb_pred, average='macro')))
print('Macro Recall: {:.4f}'.format(recall_score(y_test, gnb_pred, average='macro')))
print('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, gnb_pred, average='macro')))
print('Weighted Precision: {:.4f}'.format(precision_score(y_test, gnb_pred, average='weighted')))
print('Weighted Recall: {:.4f}'.format(recall_score(y_test, gnb_pred, average='weighted')))
print('Weighted F1-score: {:.4f}'.format(f1_score(y_test, gnb_pred, average='weighted')))
print('\n---------------Naive Bayes Classification Report ---------------\n')
print(classification_report(y_test, gnb_pred))
#print('---------------------- Naive Bayes ----------------------') # unnecessary fancy styling
