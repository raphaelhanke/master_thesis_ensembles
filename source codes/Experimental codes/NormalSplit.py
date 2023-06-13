import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("DukePPLPSB.xlsx", sheet_name="S1")

#%% 

#Splitting data
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.25)
 
train_y = train['churnInd']
test_y = test['churnInd']
 
train_x = train
train_x.pop('churnInd')
test_x = test
test_x.pop('churnInd')

#%% Making prediction

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
 
logisticRegr = LogisticRegression(random_state=5)
logisticRegr.fit(X=train_x, y=train_y)
 
test_y_pred = logisticRegr.predict(test_x)
confusion_matrix = confusion_matrix(test_y, test_y_pred)
print('Accuracy of logistic regression classifier on test set:', accuracy_score(test_y, test_y_pred))
print('F1-score of logistic regression classifier on test set:', f1_score(test_y, test_y_pred, average="macro"))
print(classification_report(test_y, test_y_pred))
 
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)