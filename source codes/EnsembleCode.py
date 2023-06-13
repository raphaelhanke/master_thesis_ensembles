# Voting Ensemble for Classification
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




dataframe = pd.read_excel("DUKEPPLPSB.xlsx", sheet_name="S1")

# Splitting to ptedictors and tarfget feature
other_features = dataframe[dataframe.columns.difference(["churnInd"])]
Y = dataframe["churnInd"].values
X = other_features.values

#%%
# cross validation
random_state_number = 7 # Standardazing random state number
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score' : make_scorer(roc_auc_score)}
kfold = model_selection.KFold(n_splits=10, random_state=random_state_number, shuffle=True)
# create the sub models
estimators = []
recall = []
f1 = []
accuracy = []
roc_auc = []

#%% Creating models

model1 = LogisticRegression(C=1000, solver = "sag", max_iter = 5000, random_state=5)
estimators.append(('logistic', model1))


model2 = DecisionTreeClassifier(criterion ="gini", max_depth = 5,
                                  min_samples_split=38,  random_state=5)
estimators.append(('dt', model2))


model3 = SVC(kernel = "linear", C=400, gamma='scale', random_state=5)  # Penalty parameter of the error term. (for the noise)
estimators.append(('svm', model3))


model4 = MLPClassifier(alpha= 0.1, hidden_layer_sizes= (11,11,11), max_iter= 1000, solver= 'lbfgs', random_state=5)  # 3 Layers with 12 neurons because we have 12 fatures; 500 iterations
estimators.append(('ann', model4))


model5 = KNeighborsClassifier(metric= 'minkowski', n_neighbors= 6, weights= 'distance')
estimators.append(('knn', model5))


model6 = GaussianNB(priors = None)
estimators.append(('nb', model6))
#

model7 = RandomForestClassifier(max_features= 'sqrt', n_estimators= 700, random_state=5)
estimators.append(('rdf', model7))
    

model8 = AdaBoostClassifier(n_estimators = 100, random_state=5)
estimators.append(('adb', model8))
    

model9 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=5)
estimators.append(('gb', model9))
    
    
#%% Loop for running the ensembles

i = 1 
# Number of classifiers in an ensemble
L = 7
for subset in itertools.combinations(estimators, L):
       ensemble = VotingClassifier(list(subset)) #Majrity voting
       results = model_selection.cross_validate(ensemble, X, Y, cv=kfold, scoring = scoring )
       print()
       print()
       print("COMBINATION " + str(i) + ":")
       print(list(subset))
       print()
       print("Accuracy:", results['test_accuracy'], "\n", "Recall:", results['test_recall'] ,"\n","F1:", results['test_f1_score'], "\n", "roc_auc:", results['test_roc_auc_score'])
       print()
       print("Accuracy mean value:", results['test_accuracy'].mean(), "\n", "Recall:", results['test_recall'].mean() , "\n", "F1 mean value:", results['test_f1_score'].mean(), "\n", "roc_auc mean value:", results['test_roc_auc_score'].mean())
       i = i + 1
    

