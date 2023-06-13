import pandas as pd
import numpy as np
from collections import Counter
from imblearn.pipeline import make_pipeline as imb
from imblearn.metrics import classification_report_imbalanced
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, \
    classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# creating dataframe
data = pd.read_excel('koreaPP50F.xlsx')

# getting info of the dataframe
data.info()


def print_result(headline, true_value, pred):
    print(headline)
    print("accuracy{}".format(accuracy_score(true_value, pred)))
    print("precision{}".format(precision_score(true_value, pred)))
    print("recall{}".format(recall_score(true_value, pred)))
    print("f1{}".format(f1_score(true_value, pred)))


#our classifier that we will be using
classifier = RandomForestClassifier

targetList = ["TARGET"]
#predictorList  = ['USEMM', 'TOTMM', 'AVG6', 'CONTACT', 'MINAP_GIGAN', 'MINAP_AMP', 'EMPTOTAL', 'CORPEV', 'JANGCNT', 'TARGET', 'CUSTGB_C', 'CUSTGB_P', 'REGION_Chungnam', 'REGION_Chungpook', 'REGION_Kangwondo', 'REGION_Unknown', 'CLAIMCNT_0', 'CLAIMCNT_1', 'CLAIMCNT_2', 'CLAIMCNT_3', 'CLAIMCNT_5', 'CLAIMCNT_6', 'CLAIMCNT_7', 'PAYMTHD_F', 'PAYMTHD_T', 'PRDCD_130', 'PRDCD_131', 'PRDCD_132', 'PRDCD_133', 'PRDCD_134', 'PRDCD_146', 'PRDCD_148', 'PRDCD_158', 'PRDCD_161', 'PRDCD_163', 'PRDCD_164', 'PRDCD_168', 'PRDCD_172', 'PRDCD_173', 'PRDCD_187', 'PRDCD_188', 'PRDCD_397', 'PRDCD_420', 'PRDCD_425', 'PRDCD_518', 'PRDCD_520', 'PRDCD_607', 'PRDCD_1001', 'PRDCD_1156', 'PRDCD_1160']  # target attribute
predictorList  = ['USEMM', 'TOTMM', 'AVG6', 'CONTACT']  # target attribute
X_train, X_test, y_train, y_test = train_test_split(data[predictorList], data[targetList] ,random_state=2, shuffle = True)

# oversampling without cross validation

# build normal model
pipeline = make_pipeline(classifier(random_state=42))
model= pipeline.fit(X_train,y_train)
prediction = model.predict(X_test)


# build model with SMOTE/ creating new samples of minority class
smote_pipeline = imb(SMOTE(random_state=4), classifier(random_state=42))
smote_model = smote_pipeline.fit(X_train,y_train)
smote_prediction = smote_model.predict(X_test)

# print information about model
print()
print("Normal data distribution: {} ".format(Counter(data["TARGET"])))
X_smote, y_smote = SMOTE().fit_sample(data[predictorList], data[targetList])
print("SMOTE data distribution: {} ".format(Counter(y_smote)))

# classification report
print(classification_report(y_test, prediction))
print(classification_report_imbalanced(y_test, smote_prediction))

print()
print("Normal Pipeline Score: {} ".format(pipeline.score(X_test, y_test)))
print("SMOTE Pipeline Score: {} ".format(smote_pipeline.score(X_test, y_test)))

print()
print_result("Normal classification ", y_test, prediction)
print()
print_result("SMOTE classification ", y_test, smote_prediction)

print()
print(prediction)


