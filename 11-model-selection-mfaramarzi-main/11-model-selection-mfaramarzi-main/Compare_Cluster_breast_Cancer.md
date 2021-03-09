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

## Summary
**To achieve leve-2 in "compare"**: 

Decision tree and SVM models (applied to classification of breast cancer severity diagnosis) were trained and optimized, and their performances were compared. For this purpose. after conducting data preparation and EDA on the dataset, samples were splitted to train and test sets. Grid search cross validation models were trained for each decision tree and SVM models and optimized paramets were found for them. Finally classification meterics were found for each modela and we compared models based on those metrics. As a result, SVM showed slightly better performance than decision tree in classification of breast cancer severity of patients. 

**To achieve level-2 in "cluster"**: 

Dataset was analyzed by k-means clustering method. We did not need optimize "k" since we already knew it is 2 because of number of classes in this dataset. Finally we calculated clustering metrics on the trained model and visualized the result of clustering in a plot. 

In assignment 9 I had used irrelevant metrics for clustering model, such as precision, recall, F1 and accuracy, however in this sumission I limited metrics to "silh" , "ARS" and "AMI". The reason those classification meterics do not work for clustering models is: By using them, we are indeed assuming that the clustering returns two distinct groups, and each of them should corresponds to our labels, which may not be the case. Actually cluster will have overlap in some extend.


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



## Loading Packages

```python
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection 
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
```

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
#convert class datatype from int to str
df1.Class= df1.Class.astype('str')
```

## EDA

```python
df1.info()
```

```python
df1.head()
```

```python
df1.shape
```

```python
sns.pairplot(data =df1, hue='Class')
```

As we can see above datapoints are not seperated and dense, which is not a good point for accurate classification and/or clusttering. However since all these plots are 2-D, we may not be able to confidently judge about that only considering pairplots, We can understand further by conducting clusttering and then evaluating relevant metrics and get help from other visualizations (e.g. by comparing clusters with ground truth labels).


### Train and Test data

```python
#numpy array of X and y data
X = df1.values[:, :9]
y = df1.values[:, 9]
```

```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X , y,test_size =.2, random_state=0)
```

```python
#shape of train data
X_train.shape
```

```python
#shape of test data
X_test.shape
```

From 683 total samples, 546(80%)  assigned to train set, and 137 (20%) assigned to the test set.


### Classification - Decision Tree

```python
#Instantiating decision tree model object
dt1 = tree.DecisionTreeClassifier(random_state=0)
```

```python
#Selected prameters for grid search optimization on decision tree model
param_grid2 = {'min_samples_leaf' : [1,2,3,4,5], 'criterion' : ['entropy', 'gini'], 'min_samples_split': [1,2,3,4,5,6]}

```

```python
#instantiating an object for GridSearchCV on dt1 model with specified param_grid
dt_opt2 = GridSearchCV(dt1,param_grid2)
```

```python
#training GridSearchCV object (dt_opt)
dt_opt2.fit(X_train,y_train)
```

```python
#results of GridSearchCV in the form of a dataframe
opt_df2 = pd.DataFrame(dt_opt2.cv_results_)
opt_df2.head()
```

```python
dt_opt2.score(X_test, y_test)
```

```python
y_pred = dt_opt2.predict(X_test)
```

```python
print(classification_report(y_test,y_pred))
```

As we can see above all metrics (precision, recall and F1) are confirming a good performance of optimized decision tree model in classification of cancer severity.

```python
df1['Class'].value_counts()
```

As we can see above, this is not a balanced dataset, but it did not have a considerable negative effect on the recall value.

```python
plot_confusion_matrix(dt_opt2,X_test, y_test )
```

AS we can see in the above confusion matrix, only 7 samples out of 137 samples could be classified accurately.


### Classification - SVM 

```python
# Instantiate  classification Support Vector Classification
svm_clf2 = svm.SVC()
```

```python
#gridsearch sets of parameters
param_grid_svm2 = {'kernel':['linear','rbf'], 'C':[.5, 1, 10]}
```

```python
#instantiate the grid search object
svm_opt2 = GridSearchCV(svm_clf2,param_grid_svm2)
```

```python
#Training the gridsearch with all sets of parameters.
svm_opt2.fit(X_train,y_train)
```

```python
#best prameters resulted from grid search CV
svm_opt2.best_params_
```

```python
#results of GridSearchCV in the form of a dataframe
opt_df_svm2 = pd.DataFrame(svm_opt2.cv_results_)
opt_df_svm2
```

Below we can see mean accuracy of prediction on the test data. This metric improved from 0.949 to 0.963 for decision tree and SVM, respectively. 

```python
# Mean accuracy 
svm_opt2.score(X_test, y_test)
```

Comparing decision tree model accuracy with SVM, we can see that accuracy improved from 94.9 to 96.3, respectively. 
Does it mean that SVM is a more relable model for classification task of this dataset? Let's do more evaluation by looking at more classification metrics.

```python
#prediction on the test set
y_pred = svm_opt2.predict(X_test)
```

```python
#classification metrics
print(classification_report(y_test,y_pred))
```

Comparing classification metrics of SVM model with decision tree model, it could be noticed that all precision, recall and F1 have improved slightly which could potentially imply better performance of SVM for this daaset. But the different is so negligible. Let's evaluate more by looking at confusion matrix and classification confidence interval. 

```python
#Plot Confusion Matrix
plot_confusion_matrix(svm_opt2,X_test, y_test )
```

Comparing SVM confusion matrix with the deciion tree one, we can notice that number of inaccurate labels decreased from 7 to 5. Considering that alone, SVM was slightly more successful than desicion tree for this classification task, however again it is a very slight difference and could be result of chance, due to size of test dataset. Because when dataset is not big enough, chance of accidental results would increase as well.


### Classification Confidence Interval


Here we want to calculate classification confidence intervals( 95% confidence level) for the accuracy of both trained classification models. Then we can evaluate how different these two classification models are. Smaller confidence interval corresponds with a more precise estimate and a larger Confidence Interval corresponds with a less precise estimate. The larger our test dataset is the smaller interval will be and consequently will have a more accurate predicted accuracy. As shown before number of test samples in this analysis is 137 (20% of whole samples). Since this number is a descent number for a test dataset, we expect to achive a relatively small interval.

```python
#Function for calculation of classification confidence with 95% confidence level
def classification_confint(acc, n):
    '''
    Compute the 95% confidence interval for a classification problem.
      acc -- classification accuracy
      n   -- number of observations used to compute the accuracy
    Returns a tuple (lb,ub)
    '''
    interval = 1.96*np.sqrt(acc*(1-acc)/n)
    lb = max(0, acc - interval)
    ub = min(1.0, acc + interval)
    return (lb,ub)
```

```python
classification_confint(svm_opt2.score(X_test,y_test),len(y_test))
```

The true classification accuracy of the model is likely between 93.2% and 99.5%.

```python
classification_confint(dt_opt2.score(X_test,y_test),len(y_test))
```

The true classification accuracy of the model is likely between 91.2% and 98.6%.


Sinnce in one hand, the accuracy intervals of these two models have considerable overlap, and in the other hand accuracy and other metrics are very close, we can conculde that models are not considered different considering their performance in prediction of breast cancer severity. 


### Clustering 


Here we want to evaluate how clustering as an unsupervised learning method, could be successful in clastering (consequently classifying) samples to to severity levels of breast cancer.

Below we can see scatter plot of two selected attributes ("Cell_Shape" and "Uniformity_Cell_Size") before clusttering. Later we will plot this again to see clusters (predicted labels) after training clusttering model.

```python
plt.scatter(df1['Cell_Shape'], df1['Uniformity_Cell_Size'])
plt.xlabel('Cell_Shape')
plt.ylabel('Uniformity_Cell_Size')
```

```python
#instantiating kmeans clustering object with two clusters
km2 = KMeans(n_clusters=2, random_state=0)
```

```python
#train kmeans model
km2.fit(X)
```

```python
#Coordinates of cluster centers. 
centroids2 = km2.cluster_centers_
centroids2
```

```python
#changing class dtype to float
df1.Class= df1.Class.astype('float')
```

As we can see above each centroid has 10 elements which stand for 10 attributes/dimentioans. But when it comes to visualization, we are only able to show centroids in 2 or 3 dimentions. That is a reason why visualization alone could not be a perfect criteria to judge performance of model and metrics would be helpful for a more mature intuition and judgment on a clusttering model.

Now let's visually compare the predicted classes using kmeans and ground truth labels. Also we visualize the location of centroids in each cluster which were calculated previsuously using 'km.cluster_centers_'.

```python
f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,4), sharey=True)
ax1.scatter(df1['Cell_Shape'], df1['Uniformity_Cell_Size'],c = km2.labels_)
ax2.scatter(df1['Cell_Shape'], df1['Uniformity_Cell_Size'],c = df1['Class'])
ax1.set_xlabel('Cell_Shape')
ax2.set_xlabel('Cell_Shape')
ax1.set_ylabel('Uniformity_Cell_Size')
ax1.set_title('k-means clustering plot')
ax2.set_title('Actual clusters')
#plotting centroids
ax1.scatter(centroids2[:, 2],centroids2[:, 1], marker = "x", s = 150)
```

As we can see above the centroid of purple cluster does not look at the center. It roots at he same reason we discussed before. This centroid is the arithmetic mean of the cluster which is in fact a 10-D shape, simplifed here in 2-D for two of attributes.

```python
#mean Silhouette Coefficient on all 9 attributes for the model trained on 9 attributes
metrics.silhouette_score(df1.values[:, :9], km2.labels_)
```

silhouette_score depends on both mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample (intra and inter similarity). The more dense and seperated clusters are, the higher silhouette_score will be (closer to 1). In this case silhouette_score = 0.597 is the result of some overlap between clusters and assigning incorrect cluster to some samples, as could be also confirmed by the above plots, as well. 

```python
metrics.adjusted_rand_score(df1['Class'],km2.labels_)
```

ARS is the measure of similarity between two clustering, then the relatively high value of ARS here confirms the similarity of predicted clusters (labels) to the ground truth labels. This result shows high number of samples were clustered (classified)correctly, and compatible with the ground truth labels. 

```python
#Adjusted Mutual Information between two clusterings.
metrics.adjusted_mutual_info_score(df1['Class'],km2.labels_)
```

AMI is correlation for categorical values. In the other words, how similar are predicted and ground truth labels for these points. In this case, AMI is 0.747 which is relatively high and shows similarity of predicted labels to ground truth labels. This actually is compatible with the other calculated metrics.
We could use AMI and ARS meterics just because we had ground truth labels for this dataset. In an unsupervised clustering we would not be able to judge based on these two metrics.
