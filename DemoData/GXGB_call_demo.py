## Step 1 Import libraries and data
# Import libraries
import geoxgboost as gx   #Imports geoxgboost
import pandas as pd
# Import data
Coords= pd.read_csv('Coords.csv' ) # Coordinates of centroid
Data = pd.read_csv('Data.csv')     # Data including GISid, X(independent variables) and  y (dependent variable)
X= Data.iloc [:, 1 : -1]           # Remove GISid and y from original data to keep only independent variables
y= Data.iloc [:, -1]               # Dependent y
VarNames = X.columns[:]            # Get variables' names. Used for labeling Dataframes

## Step 2 Hyper parameter tuning. Define initial hyperparameters for inner loop
params= {
    'n_estimators':100,     #default is 100
    'learning_rate':0.1,    #default is 0.3
    'max_depth':6,          #default is 6
    'min_child_weight':1,   #default is 1
    'gamma':0,              #default is 0
    'subsample':0.8,        #default is 1
    'colsample_bytree':0.8, #default is 1
    'reg_alpha':0,          #default is 0
     'reg_lambda':1,        #default is 1
    }
# Define search space for hyperparameteres of inner loop. A maximum of 3 hyperparameters can be tuned at the same time
Param1=None; Param2=None; Param3=None  # Set hyperparamters to None to avoid overlapping if the function runs again
Param1_Values = []; Param2_Values = []; Param3_Values = []
# Set hyperparameters and values according to the problem. Select and deselect for one or more hyperparamters
Param1='n_estimators'
Param1_Values = [100, 200, 300, 500]
Param2='learning_rate'
Param2_Values = [0.1, 0.05,0.01]
Param3='max_depth'
Param3_Values = [2,3,5,6]
#Create grid
param_grid= gx.create_param_grid(Param1,Param1_Values,Param2,Param2_Values,Param3,Param3_Values)

## Step 3 Nested CV to tune hyperparameters
params, Output_NestedCV= gx.nestedCV(X, y, param_grid, Param1, Param2, Param3, params)

## Step 4 GlobalXGBoost model
Output_GlobalXGBoost=gx.global_xgb(X,y,params)

## Step 5 Optimize Bandwidth
bw= gx.optimize_bw(X,y, Coords, params, bw_min=75, bw_max=80,step=1, Kernel='Adaptive', spatial_weights=True)

## Step 6 GXGB (Geographical-XGBoost)
Output_GXGB_LocalModel= gx.gxgb(X,y,Coords, params,bw=bw, Kernel='Adaptive', spatial_weights=True, alpha_wt_type='fixed', alpha_wt=1)

## Step 7 Predict (unseen data)
# Input data to predict
DataPredict = pd.read_csv('PredictData.csv')
CoordsPredict= pd.read_csv('PredictCoords.csv')
# predict
Output_PredictGXGBoost= gx.predict_gxgb(DataPredict, CoordsPredict, Coords, Output_GXGB_LocalModel, alpha_wt = 0.5, alpha_wt_type = 'varying')