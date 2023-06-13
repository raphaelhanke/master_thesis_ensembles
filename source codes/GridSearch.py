import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import make_pipeline as imb
from imblearn.over_sampling import SMOTE

dataframe = pd.read_excel("koreaPP50F.xlsx", sheet_name="S1")

# Splitting to ptrdictors and tarfget feature
other_features = dataframe[dataframe.columns.difference(["TARGET"])]
Y = dataframe["TARGET"].values
X = other_features.values

# cross validation
random_state_number = 7 # Standardazing random state number
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score' : make_scorer(roc_auc_score)}

kfold = model_selection.KFold(n_splits=10, random_state=random_state_number, shuffle=True)
# create the sub models
estimators = []
f1 = []
accuracy = []
roc_auc = []

# =============================================================================
# Tunning hyperparameters for LR 

#parameters = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter': [500,1000,1500], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
#clfr = LogisticRegression()
#grid = GridSearchCV(clfr, parameters,scoring='accuracy', cv=kfold)
#grid.fit(X,Y)
#print('The parameters combination that would give best accuracy is : ')
#print(grid.best_params_)
#print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# =============================================================================

# =============================================================================
# Tunning hyperparameters for DT

#parameters = {'min_samples_split':np.arange(2, 80), 'max_depth': np.arange(2,10), 'criterion':['gini', 'entropy']}
#clfr = DecisionTreeClassifier()
#grid = GridSearchCV(clfr, parameters,scoring='accuracy', cv=kfold)
#grid.fit(X,Y)
#print('The parameters combination that would give best accuracy is : ')
#print(grid.best_params_)
#print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# =============================================================================

# =============================================================================
# Tunning hyperparameters for SVM

#parameters = {"C": [0.1, 1, 10, 100, 400, 1000],
#    "kernel": ['poly', 'rbf', 'linear', 'sigmoid'],}
#clfr = SVC()
#grid = GridSearchCV(clfr, parameters,scoring='accuracy', cv=kfold)
#grid.fit(X,Y)
#print('The parameters combination that would give best accuracy is : ')
#print(grid.best_params_)
#print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# 
# =============================================================================

# =============================================================================
# Tunning hyperparameters for ANN

#parameters = {'solver': ['lbfgs', "sgd", "adam"], 'max_iter': [
#         1000], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(45, 55)}
#clfr = MLPClassifier()
#grid = GridSearchCV(clfr, parameters,scoring='accuracy', cv=kfold)
#grid.fit(X,Y)
#print('The parameters combination that would give best accuracy is : ')
#print(grid.best_params_)
#print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# 
# =============================================================================

# =============================================================================
# Tunning hyperparameters for KNN

#metrics       = ['minkowski','euclidean','manhattan'] 
#weights       = ['uniform','distance'] #10.0**np.arange(-5,4)
#numNeighbors  = np.arange(3,10)
#param_grid    = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
#clfr = KNeighborsClassifier()
#grid = GridSearchCV(clfr, param_grid=param_grid,scoring='accuracy', cv=kfold)
#grid.fit(X,Y)
#print('The parameters combination that would give best accuracy is : ')
#print(grid.best_params_)
#print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# =============================================================================

# =============================================================================
# Tunning hyperparameters for RF

#param_grid = {
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
#clfr = RandomForestClassifier()
#grid = GridSearchCV(clfr, param_grid=param_grid,scoring='accuracy', cv=kfold)
#grid.fit(X,Y)
#print('The parameters combination that would give best accuracy is : ')
#print(grid.best_params_)
#print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# =============================================================================
# Tunning hyperparameters for AB

#parameters = {'n_estimators': [100, 200, 700], 'learning_rate' : [0.01, 0.05, 0.10]}
#clfr = AdaBoostClassifier()
#grid = GridSearchCV(clfr, parameters,scoring='accuracy', cv=kfold)
#grid.fit(X,Y)
#print('The parameters combination that would give best accuracy is : ')
#print(grid.best_params_)
#print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# =============================================================================
# Tunning hyperparameters for GB

# Parameters we want to tune and their ranges
parameters = {'n_estimators': [100, 200, 700], "max_depth": [1,2,3,4,5], 'learning_rate' : [0.01, 0.05, 0.10]}
clfr = GradientBoostingClassifier()
grid = GridSearchCV(clfr, parameters,scoring='accuracy', cv=kfold)
grid.fit(X,Y)
print('The parameters combination that would give best accuracy is : ')
print(grid.best_params_)
print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
# =============================================================================

# Switch for a certain classifier
a = 0
d = 0
k = 0
l = 0
s = 0
b = 0
r = 0
ab = 0
gb = 1


if (l == 1):
    model1 = LogisticRegression( C= 1, max_iter= 500, solver= 'newton-cg', random_state=5)
    estimators.append(('logistic', model1))

if (d == 1):
    model2 = DecisionTreeClassifier()
    estimators.append(('dt', model2))

if (s == 1):
    model3 = SVC(C= 1000, kernel = 'rbf', gamma='scale', random_state=5)  # Penalty parameter of the error term. (for the noise)
    estimators.append(('svm', model3))

if (a == 1):
    model4 = MLPClassifier(alpha= 0.1, hidden_layer_sizes= (16,16,16), max_iter= 1000, solver= 'lbfgs', random_state=5)  # 3 Layers with 12 neurons because we have 12 fatures; 500 iterations
    estimators.append(('ann', model4))

if (k == 1):
    model5 = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 8, weights= 'distance')
    estimators.append(('knn', model5))

if (b == 1):
    model6 = GaussianNB(priors = None)
    estimators.append(('nb', model6))

if (r == 1):
    model7 = RandomForestClassifier(max_features= 'sqrt', n_estimators= 700, random_state=5)
    estimators.append(('rdf', model7))
        
if (ab == 1):
    model8 = AdaBoostClassifier(n_estimators = 100, random_state=5)
    estimators.append(('adb', model8))
    
if (gb == 1):
    model9 = GradientBoostingClassifier(n_estimators=700, learning_rate=0.05,
     max_depth=5, random_state=5)
    estimators.append(('gb', model9))
    
    
ensemble = VotingClassifier(list(estimators)) #Majority voting
results = model_selection.cross_validate(ensemble, X, Y, cv=kfold, scoring = scoring )
print("Accuracy:", results['test_accuracy'], "F1:", results['test_f1_score'], "roc_auc", results['test_roc_auc_score'])
print()
print("Accuracy mean value:", results['test_accuracy'].mean(), "F1 mean value:", results['test_f1_score'].mean(), "roc_auc mean value:", results['test_roc_auc_score'].mean())

