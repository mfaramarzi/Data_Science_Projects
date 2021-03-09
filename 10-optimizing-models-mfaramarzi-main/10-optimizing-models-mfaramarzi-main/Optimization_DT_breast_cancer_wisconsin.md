---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: csc310
    language: python
    name: csc310
---

## Description of Dataset
This dataset has 699 breast cancer samples. Each sample includes a code,  9 attributes and a target value as below.

1. Sample code number: id number
2. Clump Thickness: 1 - 10
3. Uniformity of Cell Size: 1 - 10
4. Uniformity of Cell Shape: 1 - 10
5. Marginal Adhesion: 1 - 10
6. Single Epithelial Cell Size: 1 - 10
7. Bare Nuclei: 1 - 10
8. Bland Chromatin: 1 - 10
9. Normal Nucleoli: 1 - 10
10. Mitoses: 1 - 10
11. Class: (2 for benign, 4 for malignant)

This dataset was extracted from the UCI machine learning repository.Data has been gathered periodically between 1989 and 1991 in seveal groups.

[Reference](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29)
## Summary
In this assignment, notebook submited in week 7, for decision tree assignment was reused to conduct optimization on its model. Unnecessary lines deleted from that notebook and focused on optimization of model. We had "mean accuracies" of decision tree model (with the manually selected hyperprameters and different size of cross validation folds (cv)) from assignment 7 submission. Then for the purpose of this assignment optimum hyperprameters (including `criterion`, `min_samples_leaf` and `min_samples_split`) were found using `GridSearchCV` function. Then, effect of changing cross validation number of folds was evaluated. Finally, several folding numbers (`cv`) in cross valication were tried and hyperpramaters were found for each `cv`. It was found that the best combination of hyerprameters for our model is `cv = 2`, `criterion : entropy`, `min_samples_leaf : 5` and `min_samples_split : 2`.  



## Loading Packages

```python
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
```

## Summary of data preparation and decision tree modeling from assignment 7.
Some data preparation was conducted on the raw dataset.Trained Models were evaluated by calculation of accuracy and visualizations. Best combinations of decision tree signatures which were tried previously was used to train decision tree models with different Train-Test split ratio. Finally results were stored in a dataframe and accuracy vs Train Test split ratio diagram was drawn. 


### Data Preparation

```python
col_names = ['Sample_code', 'Clump_Thickness ', 'Uniformity_Cell_Size', 'Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
```

```python
df1 = pd.read_csv('data/breast_cancer_wisconsin.csv', names = col_names, index_col= 'Sample_code')
```

```python
df1.head()
```

```python
#dropping any rows with "?" parameter
df1 = df1[~(df1 == '?').any(axis=1)]
```

```python
df1.Class.replace(value= ['Benign', 'Malignant'], to_replace=[2, 4], inplace=True)
```

```python
#convert class datatype from int to str
df1.Class= df1.Class.astype('str')
```

```python
# Replacing classes number by their name
df1.Class.replace(value= ['Benign', 'Malignant'], to_replace=['2', '4'], inplace=True)
```

```python
df1_percent = df1['Class'].value_counts(normalize=True).reset_index()
```

### Modeling
Train a decision tree on 20%, 30%, … , 80% of the data, using one of the training parameter combinations I tried before in the assignment 7 and worked the best.

```python
#making empty list for the dictionary "d"
ls_train_pct=[]
ls_n_train_samples=[]
ls_n_test_samples=[]
ls_train_acc=[]
ls_test_acc=[]

for train_size in range(2,9):
    train_size=train_size/10
    X_train, X_test, y_train, y_test = train_test_split (df1.values[:, :9].astype('int'),
                      df1.values[:, 9],train_size = train_size,random_state=0) 
    dt = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
    dt.fit(X_train, y_train)
    train_acc= dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    n_train_samples = X_train.shape[0]
    n_test_samples= X_test.shape[0]
    
    ls_train_pct.append(train_size)
    ls_n_train_samples.append(n_train_samples)
    ls_n_test_samples.append(n_test_samples)
    ls_train_acc.append(train_acc)
    ls_test_acc.append(test_acc)
```

```python
#final dictionary including all splits' results
d = {'train_pct': ls_train_pct, 'n_train_samples': ls_n_train_samples, 'n_test_samples': ls_n_test_samples , 'train_acc':ls_train_acc, 'test_acc': ls_test_acc}
```

```python
#Making a dataframe from dict
df_results = pd.DataFrame(data=d)
```

```python
df_results.head()
```

**Now plot the accuracies vs training percentage.**

```python
fig,ax = plt.subplots(figsize=(15,8))

ax.plot(df_results.train_pct,df_results.train_acc, label = 'Train Data')
ax.plot(df_results.train_pct,df_results.test_acc, label = 'Test Data')


ax.set_title(' Accuracies vs Training Percentage')
ax.set_xlabel('Training Percentage')
ax.set_ylabel('Accuracies')

ax.legend(fontsize= 'large')
```

According to the above plot, the accuracy of classification was not very affected by the train_test splitting ratios.


### Optimnization of Decisionb Tree Model for a More Accurate Classification

```python
#Instantiating decision tree model object
dt1 = tree.DecisionTreeClassifier(random_state=0)
```

```python
#numpy array of X and y data
X = df1.values[:, :9]
y = df1.values[:, 9]
```

To make a gridsearch optimization we need to have a ditionary of "param_grid" which enables searching over any sequence of parameter settings. when we manually tried to optimize the decision tree model we reached to the following combination of keywords:`min_samples_leaf = 3, criterion='entropy', min_samples_split= 4`. To see what combination would be recommended by gridsearch we make a list of options for each keyword.

```python
param_grid = {'min_samples_leaf' : [1,2,3,4,5], 'criterion' : ['entropy', 'gini'], 'min_samples_split': [1,2,3,4,5,6]}

```

```python
#instantiating an object for GridSearchCV on dt1 model with specified param_grid
dt_opt = GridSearchCV(dt1,param_grid)
```

```python
#training GridSearchCV object (dt_opt)
dt_opt.fit(X,y)
```

```python
#results of GridSearchCV in the form of a dataframe
opt_df = pd.DataFrame(dt_opt.cv_results_)
opt_df
```

As we can see below, there are 60 different combinations of parameters in the grid search.

```python
opt_df.shape
```

As we can see above the best test score was 0.950268 which was achive for three different combinations of keywords. `criterion : gini min_samples_leaf : 4 min_samples_split : 2,3,4`. Here score is the mean accuracy which is average of accuracy for 5 folds (cv = 5) in cross valication process.


Why test score of some rows is NaN?

```python
print(f'Best parameters {dt_opt.best_params_}')
# dt_opt.best_params_
```

Here we can see the a totally different combination of those three keywords is recommended by gridseach optimizer, in comparison with manual method. Not only `'min_samples_leaf': 4, 'min_samples_split': 2` have different values, but also criterion changed as well. 
An interesting point is that, the optimized model with `'min_samples_leaf': 4, 'min_samples_split': 2` and `cv = 5`, has lower accuracy than unoptimized model with `min_samples_leaf = 3, criterion='entropy', min_samples_split= 4` and `cv = 2`. It means that `cv` should be selected as one of the parameters in `param_grid` as well.


Why `'min_samples_split': 2` was demonstarted as the optimum value, while all `min_samples_split : 2,3,4` led to the same score?

```python
dt_opt.best_estimator_.predict(X)
```

Below we make a loop to try several cv size and get lists of best hyperprameters and corresponding mean accuracies out of each cv size(in each iteration).

```python
cv_sizes = []
best_parameters_scores = []
criterions = []
min_samples_leaves = []
min_samples_splits = []

for cv in range (2,9):
    dt_opt = GridSearchCV(dt1,param_grid, cv=cv)
    dt_opt.fit(X,y)
    cv_sizes.append(cv)
    best_parameters_scores.append(dt_opt.best_score_)
    criterions.append(dt_opt.best_params_['criterion'])
    min_samples_leaves.append(dt_opt.best_params_['min_samples_leaf'])
    min_samples_splits.append(dt_opt.best_params_['min_samples_split'])
    
```

```python
#Dictionary containing best hyperprameters of each cv_size
d = {'cv_size' : cv_sizes,'best_parameters_scores' : best_parameters_scores , 'criterions' : criterions, 'min_samples_leaves' : min_samples_leaves, 'min_samples_splits' : min_samples_splits}
```

```python
#Making a dataframe of above disctionary
df_results = pd.DataFrame(data=d)
```

As we can see below by selecting different sizes for the number of folds in a cross validation, it will lead to different values for hyperpramters. According to the below table the best mean accuracy was 0.966, and achived for `cv = 2`, with `criterion : entropy, min_samples_leaf : 5 min_samples_split : 2`. While when we used a default value (`cv = 5`), it led to best mean accuracy of 0.95 with `'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 2`.

```python
df_results.head(10)
```

By the following code we can see the optimum combination of hyperprameters which leads to the highest mean accuracy.

```python
optimized_parameters = df_results.iloc[[df_results.best_parameters_scores.argmax()]]
optimized_parameters
```
