from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_excel("koreaPP50F.xlsx", sheet_name="S1")

file = list(df.columns.values)
print()
print(file)
print()
# Create array from data values
array = df.values

X = array[:, 2:11 ] # selecting predictor features
y = array[:, 1] # selecting target attribute

# feature selection using chi square 
#sel = SelectKBest(chi2, k=3)
#sel.fit(X,y)
#a = sel.transform(X).shape
#b = sel.scores_
#print(b)

# feature selection using PCA
pca = PCA(n_components=3) 
fit = pca.fit(X)
# projection of X onto principal components
X_proj = pca.transform(X)
print(fit.components_)