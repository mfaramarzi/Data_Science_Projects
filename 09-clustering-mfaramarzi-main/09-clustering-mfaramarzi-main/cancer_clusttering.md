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

## Summary
**To earn clustering level 2:**

However, clustering is normally used as an unsupervised learning method, we used it in this assignment on a dataset including ground truth labels from assignment 7. We used the groundtruth labels for the validation process, by comparing the label of datapoints and clusters predicted for them. For this dataset we did not need to try different values of "k", since we already knew there are two clusters (two labels/possible predictions). 
In this analysis clustering grouped datapoints to two clusters which were equivalent to the two classes. In assignment 7 those classes were provided as labels for all samples. Therefore we considered clustering as an indirect way to classify datapoints to "benign" and "malignant" breast cancer.

Results were evaluated using two ways:
1. Using clustering metrics such as: 

    * silhouette_score,
    * adjusted_rand_score and
    * adjusted_mutual_info_score
    * Accuracy,
  
    
2. Visualization. For example by: 
    * Conducting an EDA (pairplots) to see the distributiona datapoints and correlation between attribuates.
    * Drawing scatter plot for both clusttered datapoints and ground truth label of datapoints and comparing them.

**To earn evaluation level 2:**
train-test splits was conducted on the dataset with different `train_size` (20%, 30%, … , 80% of whole dataset). Also metrics were evaluated to explain the function of kmeans model for each `train_size`.  


# Descriptin of Dataset
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
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import graphviz 
from sklearn.cluster import KMeans
```

## Data Preparation

```python
#We need to add headers to columns
col_names = ['Sample_code', 'Clump_Thickness ', 'Uniformity_Cell_Size', 'Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
```

```python
#Reading dataset as a Pandas dataframe
df1 = pd.read_csv('data/breast_cancer_wisconsin.csv', names = col_names, index_col= 'Sample_code')
```

```python
df1.head()
```

```python
df1.shape
```

### Cleaning Data

```python
#dropping any rows with "?" parameter
df1 = df1[~(df1 == '?').any(axis=1)]
```

## EDA

```python
sns.pairplot(df1, hue = 'Class')
```

As we can see above datapoints are not seperated and dense, which is not a good point for accurate clusttering. However since all these plots are 2-D, we may not be able to confidently judge about that only considerinf pairplots, We can understand further by conducting clusttering and then evaluating relevant metrics and get help from other visualizations (e.g. by comparing clusters with ground truth labels).


scatter plot of two selected attributes ("Cell_Shape" and "Uniformity_Cell_Size") before clusttering. Later we will plot this again to see clusters (predicted labels) after training clusttering model.

```python
plt.scatter(df1['Cell_Shape'], df1['Uniformity_Cell_Size'])
plt.xlabel('Cell_Shape')
plt.ylabel('Uniformity_Cell_Size')
```

```python
#ground truth labels
ground_truth_labels = df1['Class']
```

## Training on all attributes

```python
#instantiating kmeans clustering object with two clusters
km2 = KMeans(n_clusters=2)
km2
```

```python
#Training model on all 10 attributes
km2.fit(df1.values[:, :10])
```

```python
#Number of iterations run
km2.n_iter_
```

training process of kmeans model stopped after 7 iterations which means cluster of no datapoint would change after the seventh iteration,  in the other words, model converged at 7th iteration. By the way model is allowed to continue iterations to `max_iter=300`.

```python
#Sum of squared distances of samples to their closest cluster center
km2.inertia_
```

kmeans model by default tried 10 different centroid seed (`n_init=10`) to find the best result for clusterring considering the lowest inertia. Between those 10 trials "19420.83" was the lowest inertia for a specific centroid locations (two centroids for two clusters).
ALso, the lower the model inertia, the better the model fit. We can see that the model has very high inertia. So, it may be a potential sign that, this is not a perfect model fit to the data. We will clarify it by further analysis and using other metrics.

```python
# cluster of points as an array of zeros and ones
labels2 = km2.labels_
```

```python
#Replacing zero and one with 2 and 4, repectively, to have the same values for predicted and ground truth labels.
for n, i in enumerate(labels2):
    if i == 0:
        labels2[n] = 2
    elif i == 1:
        labels2[n] = 4
```

```python
#number of correct predictions
num_correct_labels2 = sum(ground_truth_labels == labels2)
```

```python
print("Result: %d out of %d samples were correctly labeled." % (num_correct_labels2, ground_truth_labels.size))
```

```python
#calculating accuracy of kmeans model in predicting labels
accuracy2 = num_correct_labels2/ground_truth_labels.size 
accuracy2
```

As we can see above number of correct labels is 659 (out of 683 total samples) which led to an accuracy of 0.965.


 In cases where we have a dataset labeled with classes (supervised clustering), like this dataset,  we can calculate a confusion_matrix and consequently precision, recall and f1-score.

```python
print(confusion_matrix(df1['Class'],labels2))#we should replace 0 and 1 with benign and manignant
print(classification_report(df1['Class'],labels2))
```

AS we can understand from the confusion matrix 15+9=24 samples were predicted wrongly, which is compatible with our calculation of `num_correct_labels2` before.


 **Does this clustering work better or worse than expected based on the classification performance in assignment 7?**
 
**Precision:** Improved from 0.95 (optimum DT classifier) to 0.96 (kmeans clusterring).

**Recall:** Improved from 0.93 (optimum DT classifier) to 0.96 (kmeans clusterring).

**f1-score:** Improved from 0.94 (optimum DT classifier) to 0.96 (kmeans clusterring).

However these metrics show a very slight better performance of kmeans than decision tree on this countinuous dataset, they seem to perform preety close and could be neglected. Therefore according to these metrics, both models performed well on a countinius dataset like this, with high Precision, Recall and f1-score,

```python
#Coordinates of cluster centers. 
centroids2 = km2.cluster_centers_
centroids2
```

As we can see abve each centroid has 10 elements which stand for 10 attributes/dimentioans. But when it comes to visualization, we are only able to show centroids in 2 or 3 dimentions. That is a reason why visualization alone could not be a perfect criteria to judge performance of model and metrics would be helpful for a more mature intuition and judgment on a clusttering model.


Now let's visually compare the predicted classes using kmeans and ground truth labels. Also we visualize the location of centroids in each cluster which were calculated previsuously using 'km.cluster_centers_'.

```python
f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,4), sharey=True)
ax1.scatter(df1['Cell_Shape'], df1['Uniformity_Cell_Size'],c = labels2)
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
#mean Silhouette Coefficient on all 10 attributes for the model trained on 10 attributes
metrics.silhouette_score(df1.values[:, :10], labels2)
```

silhouette_score depends on both mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample (intra and inter similarity). The more dense and seperated clusters are, the higher silhouette_score will be (closer to 1). In this case silhouette_score = 0.598 is the result of some overlap between clusters and assigning incorrect cluster to some samples, as could be also confirmed by the above plots, as well. 

```python
metrics.adjusted_rand_score(ground_truth_labels,labels2)
```

 ARS is the measure of similarity between two clustering, then the relatively high value of ARS here confirms the similarity of predicted clusters (labels) to the ground truth labels. This result is compatible with high accuracy of model, previoudsly calculated.

```python
#Adjusted Mutual Information between two clusterings.
metrics.adjusted_mutual_info_score(ground_truth_labels,labels2)
```

AMI is correlation for categorical values. In the other words, how similar are predicted and ground truth labels for these points. In this case, AMI is 0.768 which is relativelu high and shows similarity of predicted labels to ground truth labels. This actually is compatible with the other calculated metrics.
We could use AMI and ARS meterics just because we had ground truth labels for this dataset. In an unsupervised clustering we would not be able to judge based on these two metrics.


## Part 2: Test and Train Sizes



Now here we make a loop to train kmeans model using diffeent `train_size`, and find meterics them. 

```python
train_pct=[]
silh = []
ARS = []
AMI = []

for train_size in range(2,9):
    train_size=train_size/10
    X_train, X_test, y_train, y_test = train_test_split (df1.values[:, :9].astype('int'),
                      df1.values[:, 9],train_size = train_size,random_state=0) 
    km2.fit(X_train)
    
    train_pct.append(train_size)
    silh.append(metrics.silhouette_score(X_test,km2.predict(X_test)))
    ARS.append(metrics.adjusted_rand_score(y_test,km2.predict(X_test)))
    AMI.append(metrics.adjusted_mutual_info_score(y_test,km2.predict(X_test)))
```

```python
#final dictionary including all splits' results
d = {'train_pct' : train_pct, 'silh': silh, 'ARS': ARS, 'AMI': AMI}
```

```python
#Making a dataframe from calculated metrics with different train_size
df_results = pd.DataFrame(data=d)
```

```python
df_results.head(10)
```

Now we Plot `silhouette_score`, `adjusted_rand_score` and `adjusted_mutual_info_score` in one plot for all tried train_size to see the relationship between the #Plotting silhouette_score, adjusted_rand_score and adjusted_mutual_info_score in one plot for all tried `train_size`and metric scores.

```python
fig,ax = plt.subplots(figsize=(15,8))

ax.plot(df_results.train_pct,df_results.silh, label = 'silh')
ax.plot(df_results.train_pct,df_results.ARS, label = 'ARS')
ax.plot(df_results.train_pct,df_results.AMI, label = 'AMI')


ax.set_title(' Kmeans Metric Scores vs Training Samples Ratio')
ax.set_xlabel('Training Samples Ratio')
ax.set_ylabel('Metrics Score')

ax.legend(fontsize= 'large')
```

Three metrics of kmeans clustering model are drawn for different `train_size`. As could be infered from all three metrics, model deteriorates as the `train_size` increases. On the contrary, it was supposed that by increasing the `train_size` metrics improve.
