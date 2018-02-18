# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    features=X.columns
    sp=SelectPercentile(f_regression,percentile=k)
    sp.fit_transform(X,y)
    imp_features=[features[i] for i in np.argsort(sp.scores_)[::-1]]
    return imp_features[:7]
