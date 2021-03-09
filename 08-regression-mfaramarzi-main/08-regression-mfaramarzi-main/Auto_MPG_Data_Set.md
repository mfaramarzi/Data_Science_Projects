---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: csc310
    language: python
    name: csc310
---

<!-- #region -->
## Description of Dataset

The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes. We can use this dataset to predict the "mpg" of cars which is targete in this dataset. 


Target Information
1. mpg: continuous

Attribute Information:

1. cylinders: multi-valued discrete
2. displacement: continuous
3. horsepower: continuous
4. weight: continuous
5. acceleration: continuous
6. model_year: multi-valued discrete
7. origin: multi-valued discrete
8. car_name: string (unique for each instance)

[Reference](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)
<!-- #endregion -->

## Summary

To get familiar with the nature of dataset, a brief EDA (statistical and visual) was conducted initially.
After splitting dataset to train and test sets, we make a regression model to predict "mpg", using eight given  attributes. To get familiar with handling of both numerical and categorical data, this dataset was selected. Categorical data converted to dummy varibale to be able to use them in the regression model.


### Import Libraries

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
```

Choosing a name for columns

```python
col_names = ['mpg', 'cylinders', 'displacement ', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
```

```python
# seperating data with one or more space
# Assigning column names as the header of dataset
df1 = pd.read_csv('data/Auto_MPG_Data_Set/auto-mpg.data.csv', names = col_names, sep = '\s+')
```

```python
df1.head()
```

```python
# Putting target as the last col
df1 = df1[ ['cylinders', 'displacement ', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name', 'mpg']]
```

```python
df1.head()
```

```python
#dropping any rows with "?" parameter
df1 = df1[~(df1 == '?').any(axis=1)]
```

```python
#A summary of dataframe
df1.info()
```

```python
#Descriptive statistics 
df1.describe()
```

```python
#pairwise relationships between attributes of the dataset
sns.pairplot(df1)
```

Below we can see distribution of cars' horsepower

```python
sns.distplot(df1['horsepower'])
```

Below we can see a summary of correlation between "mpg" as the target value and each attribute in the form of bar charts.

```python
fig, axs = plt.subplots(nrows=2, ncols =3, figsize=(15,15))
axs[0,0].bar(df1.model_year, df1.mpg)
axs[0,0].set_title('Year vs mpg')
axs[0,0].set_xlabel('Year')
axs[0,0].set_ylabel('mpg')

axs[0,1].bar(df1.acceleration, df1.mpg)
axs[0,1].set_title('acceleration vs mpg')
axs[0,1].set_xlabel('acceleration')
axs[0,1].set_ylabel('mpg')

axs[1,0].bar(df1.weight, df1.mpg)
axs[1,0].set_title('weight vs mpg')
axs[1,0].set_xlabel('weight')
axs[1,0].set_ylabel('mpg')


axs[1,1].bar(df1.origin, df1.mpg)
axs[1,1].set_title('origin vs mpg')
axs[1,1].set_xlabel('origin')
axs[1,1].set_ylabel('mpg')


axs[1,2].bar(df1.horsepower, df1.mpg)
axs[1,2].set_title('horsepower vs mpg')
axs[1,2].set_xlabel('horsepower')
axs[1,2].set_ylabel('mpg')


axs[0,2].bar(df1.cylinders, df1.mpg)
axs[0,2].set_title('cylinders vs mpg')
axs[0,2].set_xlabel('cylinders')
axs[0,2].set_ylabel('mpg')

```

As we can see above we can visually see correlation between the "mpg" and most of attributes. For example lookinf and the "year" vs "mpg", the newer cars have higher mpg which makes sense.  In the first look, it seems weight would have the most powerful correlation with the "mpg", let's see how accurate is this prediction!


Before starting any modeling, we need to convert car_name attribute to dummy attributes.

```python
#Shape of dataframe before conversion.
df1.shape
```

```python
#Number of samples for each dummy variable will be as below.
df1['car_name'].value_counts()
```

As we can see below, totally we will have 301 dummy variables which will be replaced with the categorical variable (car_name).

```python
len(df1.car_name.unique())
```

One common solution is to convert that column to dummy columns as below. 

**Question:**  What if I disregard that categorical column instead of converting that to dummy variables. Would it reduce the score?

```python
## Converting "car_name" to dummy variables.
df1 = pd.get_dummies(df1, columns = ['car_name'])
```

```python
#shape of new dataframe
df1.shape
```

```python
#list of all of the columns in the new df1
cols = list(df1.columns) 
```

Moving "mpg" to the end of table as the target value.

```python
#Remove "mpg" from list
cols.pop(cols.index('mpg')) 
#Creating new dataframe with columns in the new order
df1 = df1[cols+['mpg']] 
```

Below we can see how dummy variables are added to the dataframe.

```python
df1.head()
```

```python
#Splitting dataset to train and test with the ratio of 0.5.
X_train, X_test, y_train, y_test = train_test_split(df1.values[:, :308], df1.values[:, 308], test_size = 0.5, random_state=0)
```

```python
# Instantiate first model (Ordinary least squares Linear Regression)
regr1 = linear_model.LinearRegression()
```

```python
#train model
regr1.fit(X_train, y_train)
```

```python
#predit on the X_test
y_pred = regr1.predict(X_test)
```

```python
#calculate r2 score
regr1.score(X_test,y_test)
#Or we could use the following command:
#r2_score(y_test,y_pred)
```

```python
mean_squared_error(y_test,y_pred)
```

As we can see above, since r2 is between 1 and zero, it is usually a better criteria to judge about the function of model, than using mean_squared_error.

```python
#learnt coefficients by reg1
regr1.coef_
```

Since we have 308 attributes after making dummy columns, we have 308 coefficient as well.

```python
#Intercept of reg1 model
regr1.intercept_
```

Now let's see how LASSO would affect the result by reducing the complexity of model.

```python
#Instantiate a lasso model with default keywords (alpha=1.0)
lasso1 = linear_model.Lasso()
#training model.
lasso1.fit(X_train, y_train)
#calculate r2
lasso1.score(X_test,y_test)
```

As we can see the r2 increased from 0.78 (for normal reg model) to 0.81 when using lasso model (with alpha=1.0).

```python
#coefficients of lasso1
lasso1.coef_
```

As we can infer from the above coefficients, model got simplified by zeroing all coefficients exept indecies 1,2,3 and 5. Between these attributes not all of them have the same weight. For example index 5,2,3  and 1 have higher weights repectively. As results, year and hourse power are the most effective variables determining mpg of a car.

```python
y_pred1 = lasso1.predict(X_test)
```

```python
mean_squared_error(y_test,y_pred1)
```

```python
lasso2 = linear_model.Lasso(alpha=.1)
lasso2.fit(X_train, y_train)
lasso2.score(X_test,y_test)
```

Above we can see the r2 even increased further when using alpha = 0.1 instead of the default alpha = 1.0. I tried lower and higher alpha but they led to lower r2, therfore, I stopped at 0.1 as the optimum alpha.

```python
lasso2.coef_
```

By redducing alpha from 1 to 0.1 R2 score increased from 0.809 to 0.820. Also "acceleration" coeficient came back to the regression model as an effective coefficient (non zero). AS observed, reducing alpha makes the losso model to involve more number of coefficients.

```python
y_pred2 = lasso2.predict(X_test)
```

```python
mean_squared_error(y_test,y_pred2)
```

**What if we only train the model on attributes with highest coefficient for each model. For example model_year and origin for lasso1 and lasso2, respectively.**


#### Training only on "model_year" attribute

```python
X_train2, X_test2, y_train2, y_test2 = train_test_split(df1.values[:, np.newaxis,5], df1.values[:, np.newaxis,308], test_size = 0.5, random_state=0)
```

```python
X_train2.shape
```

```python
lasso3 = linear_model.Lasso()
lasso3.fit(X_train2, y_train2)
lasso3.score(X_test2,y_test2)
```

```python
lasso3.coef_
```

```python
y_pred3 = lasso3.predict(X_test2)
```

```python
mean_squared_error(y_test2,y_pred3)
```

```python
plt.scatter(X_test2,y_test2, color='GREEN', label = 'Actual Datapoins')
plt.plot(X_test2,y_pred3, color='RED',label= 'Prediction')
plt.xlabel('model_year')
plt.ylabel('mpg')
plt.legend()
```

Above we can see the real datapoints and the prediction line achived by the predictions of lasso3 on the X_test2.
As we can see, points have considerable distance from the prediction line, and that has led to a poor r2.


#### Training only on "origin" attribute

```python
X_train3, X_test3, y_train3, y_test3 = train_test_split(df1.values[:, np.newaxis,6], df1.values[:, np.newaxis,308], test_size = 0.5, random_state=0)
```

```python
lasso4 = linear_model.Lasso()
lasso4.fit(X_train3, y_train3)
lasso4.score(X_test3,y_test3)
```

```python
lasso4.coef_
```

```python
y_pred4 = lasso4.predict(X_test3)
```

```python
mean_squared_error(y_test3,y_pred4)
```

Now let's have a look at the actual datapoints in comparison with the linear regression prediction line.?

```python
plt.scatter(X_test3,y_test3, color='GREEN', label = 'Actual Datapoins')
plt.plot(X_test3,y_pred4, color='RED',label= 'Prediction')
plt.xlabel('Origin')
plt.ylabel('mpg')
plt.legend()
```

Also below we conduct training on the attributes with the highest cooefficient for the normal linear regression model.Some dummy variables had higher coefficient but they could not be used for training a model and achiving an acceptable r2 (I tried). Then I selected one of the numerical viables with the highest coefficient and trained on that. 
As we saw above, wwhen we trained with lasso, we got better score than Ordinary linear regression. Also when we trained on all attributes it led to better score for both lasso and Ordinary linear regression.

```python
X_train4, X_test4, y_train4, y_test4 = train_test_split(df1.values[:, np.newaxis,0], df1.values[:, np.newaxis,308], test_size = 0.5, random_state=0)
```

```python
reg5 = linear_model.LinearRegression()
reg5.fit(X_train4, y_train4)
reg5.score(X_test4,y_test4)
```

```python
reg5.coef_
```

```python
y_pred5 = reg5.predict(X_test4)
```

```python
mean_squared_error(y_test4,y_pred5)
```

Now let's have a look at the actual datapoints in comparison with the linear regression prediction line.?

```python
plt.scatter(X_test4,y_test4, color='GREEN', label = 'Actual Datapoins')
plt.plot(X_test4,y_pred5, color='RED',label= 'Prediction')
plt.xlabel('cylinders')
plt.ylabel('mpg')
plt.legend()
```

As we can see above, the prediction line has considerable distance from the datapoints, however distances decrease for the higher number of cylinders. That could be the reason for better r2 of this model (0.62) than two models before that.


# Extra Work as practice to be complete later for portfolio


## Cross Validation


whole X and y dataset

```python
X = df1.values[:, :308]
y = df1.values[:, 308]
```

cross validation on model reg1

```python
#instead of ormal fitting we can do cross validation
regr1_cv = cross_val_score(regr1,X, y ,cv=10) 
#it returns Array of scores of the estimator for each run of the cross validation.
np.mean(regr1_cv), np.std(regr1_cv)
```

Why is it negative? DOes not make sense!

```python
regr1_cv
```

Cross-Validation on model lasso1

```python
lasso1_cv = cross_val_score(lasso1,X, y ,cv=10)
np.mean(lasso1_cv), np.std(lasso1_cv)
```

Cross-Validation on model lasso2

```python
lasso2_cv = cross_val_score(lasso2,X, y ,cv=10)
np.mean(lasso2_cv), np.std(lasso2_cv)
```

Why do we have lower scores for CV than nomrmal fits?


**Question: Does it help to normalize atributes?**


## Scaling

```python
from sklearn import preprocessing
```

```python
X_scaled = preprocessing.scale(X)
```

```python
X_scaled.mean(axis=0)
```

```python
X_scaled.std(axis=0)
```

## Normalization

```python
X_scaled2 = preprocessing.normalize(X)
```

```python
X_scaled2.mean(axis=0)
```

```python
X_scaled2.std(axis=0)
```

```python
train_stats = df1.describe()
train_stats.pop("mpg")
train_stats = train_stats.transpose()
train_stats
```

```python
type(train_stats.values)
```

```python
# def norm(x):
#   return ((x - train_stats['mean']).values / train_stats['std'].values)
# normed_data = norm(X)

```

```python
from sklearn.model_selection import KFold, ShuffleSplit
```

```python
kf = KFold(n_splits = 10)
```

```python
kf.split(X, y)
```
