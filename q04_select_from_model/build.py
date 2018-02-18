# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1:]
    model=RandomForestClassifier()
    model.fit(X,y)
    sf=SelectFromModel(model,prefit=True)
    features=X.columns[sf.get_support()==True]
    return features.tolist()
