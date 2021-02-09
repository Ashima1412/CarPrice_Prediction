# CarPrice_Prediction

# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data

df = pd.read_csv('car data.csv')
df.head()
df.shape

# Check unique values of categorical feature

print(df['Fuel_Type'].unique())
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())

# Checking missing value

df.isnull().sum()

df_new= df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
df_new.head()

# Add current Year to the dataset and calculate no of years

df_new['Current_Year'] = 2021
df_new['No_of_Year'] = df_new['Current_Year'] - df_new['Year']
df_new.head()

df_new1 = df_new[['Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type',
       'Seller_Type', 'Transmission', 'Owner', 'No_of_Year']]
df_new1.head()

# Encode Categorical feature

df_new1 = pd.get_dummies(df_new1,drop_first=True)
df_new1.head()

# Check correlation
corr = df_new1.corr()
corr

# Plot correlation

plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True)
plt.show()

# Independent and Dependent Feature
X = df_new1.iloc[:,1:]
y = df_new1.iloc[:,0]
X.head()

## Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


print(model.feature_importances_)  ## Check which feature is important

f_imp = pd.Series(model.feature_importances_,index=X.columns) # we can see easily which feature is most imporatnt
f_imp.nlargest(5).plot(kind='barh')
plt.show()

## split the data in tarin test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# To select best parameters we use HYPERPARAMETER TUNING

#no of decision trees
n_estimators = [50,100,200,300,400]
# no of feature considered at every split
max_features = ['auto','sqrt']
# max no of levels in tree
max_depth = [int(x) for x in np.linspace(5,25,num=5)]
#min no of samples required to split each node
min_samples_split = [2,5,10,15,100]
#min no of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

## Randomized Search

from sklearn.model_selection import RandomizedSearchCV
param_grid = {'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,
             'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}

rs = RandomizedSearchCV(estimator = rf,param_distributions = param_grid,verbose=2,cv=5 )

rs.fit(X_train,y_train)

y_pred = rs.predict(X_test)
y_pred1 = pd.Series(y_pred)

result = pd.DataFrame(data={'Actual':y_test,'Prediction':y_pred})
result

sns.distplot(y_test - y_pred)

plt.scatter(y_pred,y_test)

## Save the model

import pickle
file = open('random_forest_car_prediction','wb')
pickle.dump(rf,file)


