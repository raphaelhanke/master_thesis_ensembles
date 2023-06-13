import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#%% Division of the Duke dataset features by their type and function

duke = {
    "source": "Duke.xlsx",
    "sheet": "DukeDataProcessing3",
    "id_feature": "subID",
    "target_feature": "churnInd",
    "derived_features": ["LastMonthDiff", "AvgMin", "AvgDiff"],
    "categorical_features": ["plan"],
    "missing_features_mean": [],
    "missing_features_mode": []
}



#%% Division of the UCL dataset features by their type and function

ucl = {
    "source": "Dataset_UCL.xlsx",
    "sheet": "churn UCL.csv",
    "id_feature": "",
    "target_feature": "Churn",
    "derived_features": [""],
    "categorical_features": ["Area_Code2" ], 
    "binary_features": ['Intl_Plan', 'Vmail'],
    "missing_features_mean": [],
    "missing_features_mode": []
}



#%% Division of the Chile dataset features by their type and function

chile = {    
    "source": "CHILE.xlsx",
    "sheet": "BD",
    "id_feature": "ID",
    "start_date_feature" : "START_DATE",
    "end_date_feature" : "END_DATE",
    "target_feature": "CHURN",
    "derived_features": [""],
    "categorical_features": [], 
    "binary_features": [],
    "missing_features_mean": [],
    "missing_features_mode": []
    }


#%% Division of the korea dataset features by their type and function

korea = {    
    "source": "korea.xls",
    "sheet": "data1",
    "id_feature": "CUSTID",
    "target_feature": "TARGET",
    "derived_features": [""],
    "categorical_features": ["CLAIMCNT", "PRDCD"], 
    "object_features" : ['CUSTGB', 'REGION',"PAYMTHD" , "UPJONG"],
    "binary_features": ["JANGCNT"],
    "missing_features_mean": ["REV6", "AVG6", "CORPEV", "EMPTOTAL"],
    "missing_features_mode": []
    }
#%% # Creating dataframe for korea dataset and
    # Getting info of the Korea dataset

file = korea
data = pd.read_excel(file["source"], sheet_name=file["sheet"])
data.info()
file["features"] = list(data.columns.values)
data.TARGET.replace(('Y', 'N'), (1, 0), inplace=True)





#%%
# MISSING VALUES
# replace with mean
for i in file["missing_features_mean"]:
    replacement = data[i].mean()
    data[i] = data[i].fillna(replacement)
    
# replace with mode
for i in file["missing_features_mode"]:
    replacement = data[i].mode()
    data[i] = data[i].fillna(replacement)

#%%
# OUTLIERS
amount_of_deviations = 3
features = file["features"]

for i in features:
    # skip features
    if (i == file["id_feature"] or
        i == file["target_feature"] or
#        i == file["start_date_feature"] or
#        i == file["end_date_feature"] or
        #i in file["derived_features"] or
        i in file["binary_features"] or
        i in file["object_features"] or
        i in file["categorical_features"]): 
            continue
    
    mean = data[i].mean() # get mean of feature
    sd = data[i].std() # get standard deviation of feature
    lower_limit = mean - amount_of_deviations * sd # lower "acceptable" value
    upper_limit = mean + amount_of_deviations * sd # upper "acceptable" value

    # replacing outliers with "acceptable" values
    data[i] = pd.Series(
        [min(max(a, lower_limit), upper_limit)
         for a in data[i]]
    )


#%%
# CATEGORICAL TO DUMMY
# one-hot encoding
features = file["categorical_features"]

for i in features:
    data[i] = data[i].astype("object")
    data = pd.get_dummies(data)
    
features = file["object_features"]
for i in features:
    data = pd.get_dummies(data)

  

#%% seperation of id, target and other features
    
other_features = data[data.columns.difference([file["id_feature"], file["target_feature"]])]
Y = data[file["target_feature"]].values
X = other_features.values    
   

#%%Standartize - Rescale to [0:1] Range
 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X)

#Transform base set
data[data.columns.difference([file["id_feature"], file["target_feature"]])] = scaler.fit_transform(other_features)
     

#%%
# FEATURE EXTRACTION
# Recursive feature elemination ranks the features
features_to_keep = 49

model = LogisticRegression()
rfe = RFE(model, features_to_keep)
fit = rfe.fit(X_scaled, Y.ravel())

print(("Num features: %d") % fit.n_features_)
print(("Features: %s") % other_features.columns.values)
print(("Selected features: %s") % fit.support_)
print(("Feature ranking: %s") % fit.ranking_)

#%% Correlations (pairwise, prints out the combinations above the treshold)    
corr_treshold = 0.8

for i in data.columns:
    for j in data.columns:
        correlation = np.corrcoef(data[i], data[j])
        if abs(correlation[0,1]) > corr_treshold and i!=j:
            print("Correlation between " + str(i) + " and " + str(j) + " is "  + str(correlation[0,1]))

#%%
# Visualisation (full correlation matrix)
corr = data.corr() 
print(corr)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values) 


# =============================================================================
# REMOVE FEATURES CHILE

# data = data.drop(['ACTIVE_DAYS', 'ACTIVE_WEEKS','AVG_DATA_1MONTH',
#                               'AVG_DATA_3MONTH','COMPLAINT_1MONTH',
#  'COMPLAINT_1WEEK' ,'COMPLAINT_2WEEKS' ,'COMPLAINT_3MONTHS',
#  'COMPLAINT_6MONTHS', 'COUNT_CONNECTIONS_3MONTH',
# 'COUNT_SMS_INC_OFFNET_1MONTH', 'COUNT_SMS_INC_OFFNET_WKD_1MONTH',
#  'COUNT_SMS_INC_ONNET_1MONTH', 'COUNT_SMS_INC_ONNET_WKD_1MONTH' ,'COUNT_SMS_OUT_OFFNET_1MONTH'
#  ,'COUNT_SMS_OUT_OFFNET_6MONTH' ,'COUNT_SMS_OUT_OFFNET_WKD_1MONTH',
#  'COUNT_SMS_OUT_ONNET_1MONTH' ,'COUNT_SMS_OUT_ONNET_WKD_1MONTH',
# 'MINUTES_INC_ONNET__WKD_1MONTH', 'PREPAID_BEFORE', 'RECEIPT_DELAYS'], axis=1)
# data = data.drop(['AVG_DATA_1MONTH', 'AVG_DATA_3MONTH',
#                   'COMPLAINT_1WEEK' ,'COMPLAINT_2WEEKS',
#                   'COMPLAINT_3MONTHS','COUNT_CONNECTIONS_3MONTH', 'COUNT_SMS_INC_OFFNET_1MONTH', 
#                   'COUNT_SMS_INC_OFFNET_WKD_1MONTH', 'COUNT_SMS_OUT_OFFNET_1MONTH'
#  ,'COUNT_SMS_OUT_ONNET_WKD_1MONTH', 'PREPAID_BEFORE', 'RECEIPT_DELAYS'
#  ], axis=1)
# data = data.drop(['AVG_DATA_1MONTH', 'AVG_DATA_3MONTH',
#                   'COMPLAINT_1WEEK' ,'COMPLAINT_2WEEKS',
#                   'COMPLAINT_3MONTHS','COUNT_CONNECTIONS_3MONTH', 'COUNT_SMS_INC_OFFNET_1MONTH', 
#                   'COUNT_SMS_INC_OFFNET_WKD_1MONTH', 'COUNT_SMS_OUT_OFFNET_1MONTH'
#  ,'COUNT_SMS_OUT_ONNET_WKD_1MONTH', 'PREPAID_BEFORE', 'RECEIPT_DELAYS'
#  ], axis=1)  
      
# =============================================================================
# =============================================================================
# #REMOVE FEATURES KOREA
# 
# #data = data.drop(['AVG6', 'CLAIMCNT_0', 'CLAIMCNT_1', 'CLAIMCNT_17', 'CLAIMCNT_2',
# #                  'CLAIMCNT_3', 'CLAIMCNT_4', 'CLAIMCNT_5', 'CLAIMCNT_7', 'CLAIMCNT_8',
# #                 'EMPTOTAL', 'JANGCNT', 'PRDCD_1002', 'PRDCD_1164', 'PRDCD_130', 'PRDCD_131'
# # ,'PRDCD_132', 'PRDCD_141', 'PRDCD_147', 'PRDCD_152', 'PRDCD_171', 'PRDCD_176',
# # 'PRDCD_425', 'PRDCD_607', 'REGION_Busan', 'REGION_Chungnam', 'REGION_Chungpook',
# # 'REGION_Daegu', 'REGION_Daejeon', 'REGION_Gwangju', 'REGION_Gyunggido',
# # 'REGION_Incheon', 'REGION_Jeju', 'REGION_Jeonnam', 'REGION_Jeonpook',
# # 'REGION_Kangwondo', 'REGION_Kyungnam', 'REGION_Kyungpook', 'REGION_Seoul',
# # 'REGION_Ulsan', 'REGION_Unknown', 'REV6', 'USEMM'
# # ], axis=1)
# 
data = data.drop(['CLAIMCNT_17','CLAIMCNT_4','CLAIMCNT_8',
                  'PRDCD_1002', 'PRDCD_1164'
  ,'PRDCD_141', 'PRDCD_147', 'PRDCD_152', 'PRDCD_171', 'PRDCD_176',
  'REGION_Busan',
  'REGION_Daegu', 'REGION_Daejeon', 'REGION_Gwangju', 'REGION_Gyunggido',
  'REGION_Incheon', 'REGION_Jeju', 'REGION_Jeonnam', 'REGION_Jeonpook',
  'REGION_Kyungnam', 'REGION_Kyungpook', 'REGION_Seoul',
  'REGION_Ulsan', 'REV6'
  ], axis=1)
# =============================================================================



data.to_excel('C:/Users/Heg/Desktop/DataSets2/koreaPP50F.xlsx', sheet_name="S1")