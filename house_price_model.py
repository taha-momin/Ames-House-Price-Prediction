import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import joblib

df=pd.read_csv('ames_housing.csv')

plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.title('Gr Liv Area vs SalePrice')
plt.show()
plt.scatter(df['OverallQual'], df['SalePrice'])
plt.title('Overall Quality vs SalePrice')
plt.show()
plt.scatter(df['GarageCars'],df['GarageArea'])
plt.title('Garage Cars vs Garage Area')
plt.show()

plt.boxplot(data=df,x='neighborhood',y='SalePrice')
plt.title('Sale Price by Neighborhood')
plt.show()
plt.boxplot(data=df,x='OverallQual',y='SalePrice')
plt.title('Sale Price by Overall Quality')
plt.show()

corr=df.select_dtypes(include=np.number).corr()['SalePrice'].sort_values(ascending=False)
print(corr.head(15))
print(corr.tail(15))

top_features=df.select_dtypes(include=np.number).corr()['SalePrice'].abs().sort_values(ascending=False).head(15).index
sns.heatmap(df[top_features].corr(),annot=True,cmap='coolwarm')
plt.show()

X=df[top_features].dropna()
vif=pd.DataFrame()
vif['Features']=X.columns
vif['VIF']=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif.sort_values(by='VIF',ascending=False))
