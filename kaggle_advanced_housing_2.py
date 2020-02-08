'''https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview'''

"""This time, let's see how lightGBM stacks up against XGBoost"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns

# Read the data
X = pd.read_csv('D:/code/Data/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test_full = pd.read_csv('D:/code/Data/house-prices-advanced-regression-techniques/test.csv', index_col='Id')


# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 100 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)





#Keep track of hyper paramaters as we iterate through
hyper=pd.DataFrame()
hyper_min=pd.DataFrame()





#iterating through the hyperparameters
'''paramaters to iterate:
    n_estimators
    learning_rate
    early_stopping_rounds
    tree_depth
    
    '''

data=[]

for est in range(10000,10001,1000):
    for learn in np.arange(.099,.100,.001):
        for early in range(99,100):
          
            my_model = XGBRegressor(objective ='reg:squarederror',n_estimators=est, learning_rate=learn, n_jobs=4)
            my_model.fit(X_train, y_train, early_stopping_rounds=early, eval_set=[(X_valid, y_valid)], verbose=2)
            predictions = my_model.predict(X_valid)
            mae=mean_absolute_error(predictions, y_valid)
            
            data.append({'estimators':est,'learning_rate':learn,'early_stopping_rounds':early,'mae':mae})
            print(f'{est} {learn} {early} {mae}')
            
temp_hyper=pd.DataFrame(data)
hyper=hyper.append(temp_hyper, ignore_index=True)
temp_hyper_min=temp_hyper[temp_hyper.mae == temp_hyper.mae.min()]
hyper_min=hyper_min.append(temp_hyper_min, ignore_index=True)

hyper.to_csv(r'D:/code/Data/hyper1.csv')
hyper_min.to_csv(r'D:/code/Data/hyper_min1.csv')
temp_hyper_min.to_csv(r'D:/code/Data/temp_hyper_min1.csv')





#Look at the stuff we made
print(hyper)
print(hyper_min)
print(temp_hyper_min)
print(temp_hyper_min.describe())




#train a model and make predictions based on our lowest MAE from the grid search above
my_model = XGBRegressor(objective ='reg:squarederror',n_estimators=10000, learning_rate=.102, n_jobs=1)
my_model.fit(X_train, y_train, early_stopping_rounds=99, eval_set=[(X_valid, y_valid)], verbose=False)
preds_test = my_model.predict(X_test)





#save our predictions in submission format

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})

output.to_csv(r'D:/code/Data/house-prices-advanced-regression-techniques/house_sub_3.csv',index=False, header =1)


