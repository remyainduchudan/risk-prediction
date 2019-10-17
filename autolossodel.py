#import dependencies
import sklearn 
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

#read dataset
df = pd.read_csv("autoloss.csv")
include = ['Age','GENNUM','MATNUM','CLASS'] #ONLY 4 FEATURES
df_ = df[include]

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)


# K nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
dependent_variable = "CLASS"
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
k = 9
lr = KNeighborsClassifier(n_neighbors = k)
lr.fit(x,y)

# Save your model
from sklearn.externals import joblib
joblib.dump(lr, 'autolossmodel.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('autolossmodel.pkl')

# Saving the data columns from training
autolossmodel_columns = list(x.columns)
joblib.dump(autolossmodel_columns, 'autolossmodel_columns.pkl')
print("Models columns dumped!")
