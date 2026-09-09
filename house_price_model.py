import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import joblib

df=pd.read_csv('ames_housing.csv')

# plt.scatter(df['GrLivArea'], df['SalePrice'])
# plt.title('Gr Liv Area vs SalePrice')
# plt.show()
# plt.scatter(df['OverallQual'], df['SalePrice'])
# plt.title('Overall Quality vs SalePrice')
# plt.show()
# plt.scatter(df['GarageCars'],df['GarageArea'])
# plt.title('Garage Cars vs Garage Area')
# plt.show()

# plt.boxplot(data=df,x='Neighborhood',y='SalePrice')
# plt.title('Sale Price by Neighborhood')
# plt.show()
# plt.boxplot(data=df,x='OverallQual',y='SalePrice')
# plt.title('Sale Price by Overall Quality')
# plt.show()

# corr=df.select_dtypes(include=np.number).corr()['SalePrice'].sort_values(ascending=False)
# print(corr.head(15))
# print(corr.tail(15))

# top_features=df.select_dtypes(include=np.number).corr()['SalePrice'].abs().sort_values(ascending=False).head(15).index
# sns.heatmap(df[top_features].corr(),annot=True,cmap='coolwarm')
# plt.show()

# X=df[top_features].dropna()
# vif=pd.DataFrame()
# vif['Features']=X.columns
# vif['VIF']=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif.sort_values(by='VIF',ascending=False))

#HANDLING MISSING VALUES AND OUTLIERS >>>>>

df=df[df['GrLivArea']<4500].reset_index(drop=True)

none_cols=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageFinish',
           'GarageQual','GarageCond','GarageType','BsmtExposure','BsmtFinType2',
           'BsmtFinType1','BsmtCond','BsmtQual']
for col in none_cols:
    df[col]=df[col].fillna('None')


df['LotFrontage']=df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df['MasVnrArea']=df['MasVnrArea'].fillna(0)
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
df['MasVnrType']=df['MasVnrType'].fillna('None')
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])

df['SalePrice_log']=np.log1p(df['SalePrice'])

zero_heavy_cols=['PoolArea','3SsnPorch','LowQualFinSF','MiscVal','ScreenPorch','BsmtHalfBath','BsmtFinSF2','EnclosedPorch']
for col in zero_heavy_cols:
    df[f'has_{col}']=(df[col]>0).astype(int)

#ENCODING >>>>>>>>

ordinal_cols=['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu',
              'GarageQual','GarageCond','PoolQC']

nominal_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                'CentralAir', 'Electrical', 'Functional', 'GarageType', 'PavedDrive',
                'SaleType', 'SaleCondition', 'Alley', 'Fence', 'MiscFeature']

skewed_cols = ['LotArea', 'LotFrontage', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']
extra_ordinal = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish']

quality_order=['None','Po','Fa','TA','Gd','Ex']
encoder=OrdinalEncoder(categories=[quality_order]*len(ordinal_cols))
df[ordinal_cols]=encoder.fit_transform(df[ordinal_cols])

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df[nominal_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(nominal_cols))
df = pd.concat([df.drop(columns=nominal_cols).reset_index(drop=True), encoded_df], axis=1)

robust_scaler=RobustScaler()
df[skewed_cols]=robust_scaler.fit_transform(df[skewed_cols])

df['BsmtExposure']=OrdinalEncoder(categories=[['None','No','Mn','Av','Gd']]).fit_transform(df[['BsmtExposure']])
for col in ['BsmtFinType1','BsmtFinType2']:
    df[col]=OrdinalEncoder(categories=[['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ']]).fit_transform(df[[col]])
df['GarageFinish']=OrdinalEncoder(categories=[['None','Unf','RFn','Fin']]).fit_transform(df[['GarageFinish']])


y=df['SalePrice_log']
X=df.drop(columns=['SalePrice','SalePrice_log'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

X_train[skewed_cols]=robust_scaler.fit_transform(X_train[skewed_cols])
X_test[skewed_cols]=robust_scaler.transform(X_test[skewed_cols])

excluded_from_scaling=skewed_cols+ordinal_cols+nominal_cols+extra_ordinal+list(encoded_df.columns)
remaining_numeric=[col for col in X_train.select_dtypes(include=[np.number]).columns if col not in excluded_from_scaling]

standard_scaler=StandardScaler()
X_train[remaining_numeric]=standard_scaler.fit_transform(X_train[remaining_numeric])
X_test[remaining_numeric]=standard_scaler.transform(X_test[remaining_numeric])


#TRAINING MODELS >>>>>>>>

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr=lr.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,y_pred_lr))
r2=r2_score(y_test,y_pred_lr)


en=ElasticNet(alpha=0.01,l1_ratio=0.5,random_state=20)
en.fit(X_train,y_train)
y_pred_en=en.predict(X_test)

rmse_final=np.sqrt(mean_squared_error(y_test,y_pred_en))
r2_final=r2_score(y_test,y_pred_en)

rf=RandomForestRegressor(n_estimators=300,max_depth=20,min_samples_leaf=2,min_samples_split=5,random_state=20)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)

rmse_best=np.sqrt(mean_squared_error(y_test,y_pred_rf))
r2_best=r2_score(y_test,y_pred_rf)


# param_grid = {
#     'alpha': [0.0001, 0.0005, 0.001, 0.005,0.01],
#     'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9,1.0]
# }
# grid=GridSearchCV(estimator=ElasticNet(random_state=20), param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
# grid.fit(X_train, y_train)
# print(f'Best parameters: {grid.best_params_}')
# print(f'Best R2 score: {grid.best_score_}')

# best_en=grid.best_estimator_
# y_pred_best_en=best_en.predict(X_test)
# rmse_best_en=np.sqrt(mean_squared_error(y_test,y_pred_best_en))
# r2_best_en=r2_score(y_test,y_pred_best_en)
# print(f'Best Elastic Net - RMSE: {rmse_best_en}, R2: {r2_best_en}')

print(f"{'Model':<25}{'RMSE':<10}{'R2':<10}")
print(f"{'Linear Regression':<25}{rmse:<10.4f}{r2:<10.4f}")
print(f"{'Random Forest (tuned)':<25}{rmse_best:<10.4f}{r2_best:<10.4f}")
print(f"{'ElasticNet (final)':<25}{rmse_final:<10.4f}{r2_final:<10.4f}")