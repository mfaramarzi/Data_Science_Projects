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

Malignant'cancers. This dataset was extracted from the UCI machine learning repository.Data has been gathered periodically between 1989 and 1991 in seveal groups

[Reference](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29)
## Summary
accordingly some data preparation was conducted to give the raw dataset a more appropriate configuration. Dataset contained some non-numerical values which was considered by cleaning dataset. Visual and Statistical EDA was conducted on the data.Trained Models were evaluated by different metrics and visualizations. Different combinations of decision tree signatures was applied to evaluate their influence on the meterics. Different Train-Test split ratio was tried and inally results were stored in a dataframe. 


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
import matplotlib.pyplot as plt
import graphviz 
```

## Data Preparation


We need to add headers to columns

```python
col_names = ['Sample_code', 'Clump_Thickness ', 'Uniformity_Cell_Size', 'Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
```

```python
df1 = pd.read_csv('breast_cancer_wisconsin.csv', names = col_names, index_col= 'Sample_code')
```

```python
df1.head()
```

Below we can see list of attributes (variables) and target values.

```python
print(df1.columns)
```

```python
#Shape of dataframe
df1.shape
```

### Cleaning Data

```python
#dropping any rows with "?" parameter
df1 = df1[~(df1 == '?').any(axis=1)]
```

```python
df1.shape
```

we cleaned data by deleting 16 rows including "?" character. 

```python
df1.dtypes
```

As we can see above not all attributes are numerical


## Part 1


**Question: What is a classification Model?**

A classification model is a type of model in datascience that predicts a class type outcome. A classification model uses attributes of dataset to predict a class category. 

**Question: Why a decision tree is a reasonable model to try for this data?**

In this dataset we have 699 samples with 9 attributes and a column of labels/targets. The target value is severity of breast cancer and if it is either Benign or Malignant. We can use a decision tree model to find logics to recognize the severity of cancer considering the number in 9 different attributes. I think decision tree works for this data, since we can categorize samples using "if" and "then" statements which are in the natures off a decision tree model.


## EDA


Let's draw a pairplot for the dataset to observe the correlation between attributes

```python
sns.pairplot(df1, hue = 'Class')
```

as we can see above datapoints are not forming a blub shape as was preferable for GNB model. Also, datapoints are scattered.


### Satistical Summary of Dataset

```python
df1.describe()
```

mean of variables for each class

```python
df1.Class.replace(value= ['Benign', 'Malignant'], to_replace=[2, 4], inplace=True)
```

We can group data by their class and find the average values for each feature as below

```python
df1_mean = df1.groupby('Class').mean()
df1_mean
```

As we can see above attributes are distinct between two classes and potentially could be classified using decision tree model.


drawing bar plots of mean of each features for each class

```python
#plotting style
plt.style.use('seaborn')
#defining subplot
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7), sharey=True)#sharing y axis to plot them with the same scale
#defining each plot
ax1.bar(df1_mean.columns, df1_mean.loc["Benign"])
ax2.bar(df1_mean.columns, df1_mean.loc["Malignant"])
#title and labels
ax1.set_title("Benign")
ax1.set_xlabel('Attribute')
ax1.set_ylabel('value')
ax2.set_title("Malignant")
# fits in to the figure area.
plt.tight_layout()


#how to change the size of font?

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 1}

# plt.rc('font', **font)
# plt.rcParams.update({'font.size': 2})

```

Question: How can I make a better x-axis label in the above plots?


### Statistical Summary for each class

```python
df1.groupby('Class').describe().T
```

### More Data Preparation

```python
#convert class datatype from int to str
df1.Class= df1.Class.astype('str')
```

```python
# Replacing classes number by their name
df1.Class.replace(value= ['Benign', 'Malignant'], to_replace=['2', '4'], inplace=True)
```

```python
df1.iloc[::5]
```

As we canvalues see above class values are replaced accordingly.


Let's see how many percent samples assigns to each class.

```python
df1_percent = df1['Class'].value_counts(normalize=True).reset_index()
df1_percent
```

It could also be nicely illustrated in a pie chart as below.

```python
#choosing a style for the plot
plt.style.use('fivethirtyeight')
#figure size
plt.figure(figsize=(8,8)) 
#draw pie chart of severety distribution
plt.pie(df1_percent.Class,  labels=['Benign', 'Malignant'],  autopct = '%1.1f%%')#he label will be placed inside the wedge
```

```python
#Splitting data to train and test sets (each 50%)
X_train, X_test, y_train,  y_test = train_test_split(df1.values[:,:9].astype('int'),df1.values[:,9],
                                                     train_size=.5, random_state=0)# we assign a number to random_state to get the same samples everytime we run 
```

**Decision tree model with default signatures**

```python
#decision tree with the default parameters on 50% of the data
dt1 = tree.DecisionTreeClassifier(random_state=0)
#By defualt: max_depth=None, min_samples_split=2, min_samples_leaf=1
```

```python
#TRaining model
dt1.fit(X_train, y_train)
```

```python
#Build a text report showing the rules of this decision tree.
print(tree.export_text(dt1))
```

```python
#Plot the decision tree
tree.plot_tree(dt1, fontsize=5)
plt.savefig('tree_high_dpi', dpi=100)
```

As we can see in the above tree, it has six levels. Also the minimum number of samples in a leaf is 1, which makes sense, because of using the default values for the  keywords. ( ```min_samples_split=2```,```min_samples_leaf=1```). ALso we can see that gini score of last leaves are all zero.


**We can also export the tree in Graphviz format using the export_graphviz exporter**

```python
dot_data = tree.export_graphviz(dt1, out_file='dt1', filled=True)
```

In the exported graphvize visualized tree, classes are demonstrated by two different colors in each node. The more porobabality a node has to have one class, corresponding color for that class will be more intense. Also nodes with equal chance are white.

```python
dt1.tree_.max_depth
```

```python
# graph = graphviz.Source(dot_data) 
# graph
```

**Question: How does a DT model decides about the conditions limits? e.g feature_1 <= 3.50**

Using gini impurity following steps below:

* it sorts the attributes lowest to highest.
* calculates the average values for all adjacent samples.
* Calculates the impurity values for each average value.
* We choose the minimum impurity value.


```python
#accuracy on the given train data and labels
dt1.score(X_train, y_train)
```

```python
#accuracy on the given test data and labels
dt1.score(X_test, y_test)
```

By comparing the accuracy of prediction on the train and test data we can interfer that model has better performance on tran set, as expected because some level of memorizing data. Maybe there is overfitting in some level which could be prevented by assigning a different ```min_samples_split```, ```max_depth```  and```min_samples_leaf```.


## Metrics
### Precision vs  Recall Curve

```python
disp = plot_precision_recall_curve(dt1, X_test, y_test)
```

In the above recision-recall curve shows a high area because both precisionand recall have high values. High value of precision is due to low false possitive (12 samples for each class as shown in the below confusion matrix) , on the other hand, high recall is because of low false negative (12 samples for each class as shown in the below confusion matrix). As we have high values (close to 1) for both precision and recall, we can infer that model is predicting accurate results and also ais able to find most of all possitive results.


finding the predicted y for test set.

```python
y_pred1 = dt1.predict(X_test)
```

### Confusion Matrix

```python
confusion_matrix(y_test, y_pred1)
```

```python
plot_confusion_matrix(dt1,X_test, y_test )
```

Let's find the precision


### Precision and Recall

```python
precision_score(y_test, y_pred1, average='macro')
```

If we set ```average='macro'``` as above, it Calculate metrics for each label, and find their unweighted mean.  This does not take label imbalance into account.


If we set ```average=None```, the scores for each class are returned as below.

```python
precision_score(y_test, y_pred1, average=None)
```

```python
recall_score(y_test, y_pred1, average='macro')
```

### F1 scores


F-1 score is a more comprehensive parameter which considers both precision and recall to judge the function of model. As we can see below since precision and recall had relatively simailar values, their f-1 is almost the same as them which makes sense considering their formua.

```python
f1_score(y_test, y_pred1, average='macro')
```

Because the tree grew fully to reach the lowest impurity for each leaf, a couple of leaves had a 1 or two number of samples which leads to overfitting. We can infer this overfitting from the relatively big difference between train and test accuracies as well. In general there should be a balance between bias and variance to reach to the highest accuracy of test dataset.


**Repeat with the entropy criterion**

```python
dt1_entropy = tree.DecisionTreeClassifier(random_state=0, criterion='entropy')
```

```python
dt1_entropy.fit(X_train, y_train)
```

```python
dt1_entropy.score(X_train, y_train)
```

```python
dt1_entropy_test_acc = dt1_entropy.score(X_test, y_test)
dt1_entropy_test_acc
```

```python
disp = plot_precision_recall_curve(dt1_entropy, X_test, y_test)
```

```python
y_pred1_entropy = dt1_entropy.predict(X_test)
```

```python
precision_score(y_test, y_pred1_entropy, average='macro')
```

```python
recall_score(y_test, y_pred1_entropy, average='macro')
```

```python
f1_score(y_test, y_pred1_entropy, average='macro')
```

AS we can see above using entropy criterion increased prediction accuracy on both train and test datasets. In this level of accuracy accuracy improvement from 0.932 to 0.956 is considerable. So entropy worked better than gini.


## Part 2: DT parameters


#### Now let's evaluate the effect of changing  ```min_samples_leaf ``` in ```DecisionTreeClassifier()``` by changing that from 5 leaf to 10 leaf

```python
dt2 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 10)
```

```python
dt2.fit(X_train, y_train)
```

```python
#accuracy on the given test data and labels
dt2.score(X_test, y_test)
```

we can see that changing the  ```min_samples_leaf ``` from 5 to 10 decreased the accuracy from 0.933 to 0.939. It is not a considerable change, but anyway it was beneficial to increase min_samples_leaf.


```python
disp = plot_precision_recall_curve(dt2, X_test, y_test)
```

```python
y_pred2 = dt2.predict(X_test)
```

```python
precision_score(y_test, y_pred2, average='macro')
```

```python
recall_score(y_test, y_pred2, average='macro')
```

```python
f1_score(y_test, y_pred2, average='macro')
```

#### Now let's evaluate the effect of changing  ```min_samples_leaf ``` in ```DecisionTreeClassifier()``` by changing that to 3 leaf

```python
dt3 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3)
```

```python
dt3.fit(X_train, y_train)
```

```python
#accuracy on the given test data and labels
dt3.score(X_test, y_test)
```

```python
disp = plot_precision_recall_curve(dt3, X_test, y_test)
```

```python
y_pred3 = dt3.predict(X_test)
```

```python
precision_score(y_test, y_pred3, average='macro')
```

```python
recall_score(y_test, y_pred3, average='macro')
```

```python
f1_score(y_test, y_pred3, average='macro')
```

we can see that changing the  ```min_samples_leaf ``` to 3, increased the accuracy to 0.956. It is a progress. 

**Let's see if it even further improves if we decrease the ```min_samples_leaf``` to 2 and 1.**


```python
dt4 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 2)
```

```python
dt4.fit(X_train, y_train)
```

```python
#accuracy on the given test data and labels
dt4.score(X_test, y_test)
```

```python
disp = plot_precision_recall_curve(dt4, X_test, y_test)
```

```python
y_pred4 = dt4.predict(X_test)
```

```python
precision_score(y_test, y_pred4, average='macro')
```

```python
recall_score(y_test, y_pred4, average='macro')
```

```python
f1_score(y_test, y_pred4, average='macro')
```

```python
dt5 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 1)
```

```python
dt5.fit(X_train, y_train)
```

```python
#accuracy on the given test data and labels
dt5.score(X_test, y_test)
```

```python
disp = plot_precision_recall_curve(dt5, X_test, y_test)
```

```python
y_pred5 = dt5.predict(X_test)
```

```python
precision_score(y_test, y_pred5, average='macro')
```

```python
recall_score(y_test, y_pred5, average='macro')
```

```python
f1_score(y_test, y_pred5, average='macro')
```

we can not see a clear correlation between number of minimum leaf and accuracy. 

**Now let's keep ```min_samples_leaf=3``` constant as it led to the highest accuracy (0.938) on the test data, and change the dataset split ratio by changing the assigned ratio to ```train_size``` keyword in ```train_test_split()```and see how that affects the accuracy.**

```python
X_train2, X_test2, y_train2, y_test2 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = 0.7,random_state=0) 
```

```python
dt6 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3)
```

```python
dt6.fit(X_train2, y_train2)
```

```python
dt6.score(X_test2, y_test2)
```

we can see that changing train_size from 0.5 to 0.7 did not change the accuracy considerably, since with the same min_samples_leaf (3) they had 0.9385 and 0.9365 accuracy respectively.

```python
disp = plot_precision_recall_curve(dt6, X_test2, y_test2)
```

```python
y_pred6 = dt6.predict(X_test2)
```

```python
precision_score(y_test2, y_pred6, average='macro')
```

```python
recall_score(y_test2, y_pred6, average='macro')
```

```python
f1_score(y_test2, y_pred6, average='macro')
```

Now let's again change the min_samples_leaf back to 5, to see if ```min_samples_leaf=3``` leads to the highest accuracy for ```train_size = 0.7``` as it did for ```train_size = 0.5``` or not.

```python
dt7 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 5)
```

```python
dt7.fit(X_train2, y_train2)
```

```python
dt7.score(X_test2, y_test2)
```

```python
disp = plot_precision_recall_curve(dt7, X_test2, y_test2)
```

```python
y_pred7 = dt7.predict(X_test2)
```

```python
precision_score(y_test2, y_pred7, average='macro')
```

```python
recall_score(y_test2, y_pred7, average='macro')
```

```python
f1_score(y_test2, y_pred7, average='macro')
```

we can see that when we use ```min_samples_leaf=3``` the optimum number fot ```min_samples_leaf``` would change as well.


 **Now let's try a entropy criterion as the function to measure the quality of a split**

```python
dt8 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy')
```

```python
dt8.fit(X_train2, y_train2)
```

```python
dt8.score(X_test2, y_test2)
```

```python
dt8.score(X_train2, y_train2)
```

By changing the criterion from gini to entropy, accuracy increase from 0.9365 to 0.9414.  So changing the criterion was not a very effective way to improve the model. But still in that level of accuracy even a tiny incease is precious.

As we can see above, accuracy of prediction on train samples is higher than test samples,so,maybe model is overfitting a little bit. We can check that by defining a maximum depth. Let's see what the max depth was in dt8.

```python
dt8.tree_.max_depth
```

```python
disp = plot_precision_recall_curve(dt8, X_test2, y_test2)
```

```python
y_pred8 = dt8.predict(X_test2)
```

```python
precision_score(y_test2, y_pred8, average='macro')
```

```python
recall_score(y_test2, y_pred8, average='macro')
```

```python
f1_score(y_test2, y_pred8, average='macro')
```

**Now let's try a different maximum_depth and check how this prameter affects the accuracy.**

By controlling the maximum depth we can control overfitting. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples

```python
dt9 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='gini', max_depth=5)
```

```python
dt9.fit(X_train2, y_train2)
```

```python
dt9.score(X_test2, y_test2)
```

As we can see above decreasing the maximum depth did not improve accuracy, then the max  depth is not a factor of overfitting.

```python
disp = plot_precision_recall_curve(dt9, X_test2, y_test2)
```

```python
y_pred9 = dt9.predict(X_test2)
```

```python
precision_score(y_test2, y_pred9, average='macro')
```

```python
recall_score(y_test2, y_pred9, average='macro')
```

```python
f1_score(y_test2, y_pred9, average='macro')
```

**Now let's see how ```min_samples_split``` affects the accuracy**

```python
dt10 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy',max_depth=5, min_samples_split= 4)
```

```python
dt10.fit(X_train2, y_train2)
```

```python
dt10.score(X_test2, y_test2)
```

```python
disp = plot_precision_recall_curve(dt10, X_test2, y_test2)
```

```python
y_pred10 = dt10.predict(X_test2)
```

```python
precision_score(y_test2, y_pred10, average='macro')
```

```python
recall_score(y_test2, y_pred10, average='macro')
```

```python
f1_score(y_test2, y_pred10, average='macro')
```

As we can see change of ```min_samples_split``` from default value (2) to 4 did not change accuracy considerably. It was chnaged from 0.936 to 0.941.


## Part 3: Test and Train Sizes


Train a decision tree on 20%, 30%, … , 80% of the data, using one of the training parameter combinations you tried above and explain why you chose the one you chose.

```python
train_size20 = 0.2
```

```python
X_train3, X_test3, y_train3, y_test3 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = train_size20,random_state=0) 
```

```python
dt11 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
```

I got different accuracies in each run, then I could not compare models with different hyper pramaters, because they changed in every run.

```python
dt11.fit(X_train3, y_train3)
```

```python
dt11_train_acc = dt11.score(X_train3, y_train3)
dt11_train_acc
```

```python
dt11_test_acc = dt11.score(X_test3, y_test3)
dt11_test_acc
```

```python
#Making a dict of data
d_20 = {'train_size':train_size20,
     'n_train_samples': X_train3.shape[0],
     'n_test_samples': X_test3.shape[0],
        'train_acc': dt11_train_acc,
        'test_acc' : dt11_test_acc}
```

```python
#Assigning train size to an object
train_size30 = 0.3
```

```python
X_train4, X_test4, y_train4, y_test4 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = train_size30,random_state=0) 
```

```python
dt12 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
```

```python
dt12.fit(X_train4, y_train4)
```

```python
dt12_train_acc = dt12.score(X_train4, y_train4)
dt12_train_acc
```

```python
dt12_test_acc = dt12.score(X_test4, y_test4)
dt12_test_acc
```

```python
d_30 = {'train_size':train_size30,
     'n_train_samples': X_train4.shape[0],
     'n_test_samples': X_test4.shape[0],
        'train_acc': dt12_train_acc,
        'test_acc' : dt12_test_acc}
```

```python
train_size40 = 0.4
```

```python
X_train5, X_test5, y_train5, y_test5 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = train_size40,random_state=0) 
```

```python
dt13 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
```

```python
dt13.fit(X_train5, y_train5)
```

```python
dt13_train_acc = dt13.score(X_train5, y_train5)
dt13_train_acc
```

```python
dt13_test_acc = dt13.score(X_test5, y_test5)
dt13_test_acc
```

```python
d_40 = {'train_size':train_size40,
     'n_train_samples': X_train5.shape[0],
     'n_test_samples': X_test5.shape[0],
        'train_acc': dt13_train_acc,
        'test_acc' : dt13_test_acc}
```

```python
train_size50 = 0.5
```

```python
X_train6, X_test6, y_train6, y_test6 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = train_size50,random_state=0) 
```

```python
dt14 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
```

```python
dt14.fit(X_train6, y_train6)
```

```python
dt14_train_acc = dt14.score(X_train6, y_train6)
dt14_train_acc
```

```python
dt14_test_acc = dt14.score(X_test6, y_test6)
dt14_test_acc
```

```python
d_50 = {'train_size':train_size50,
     'n_train_samples': X_train6.shape[0],
     'n_test_samples': X_test6.shape[0],
        'train_acc': dt14_train_acc,
        'test_acc' : dt14_test_acc}
```

```python
train_size60 = 0.6
```

```python
X_train7, X_test7, y_train7, y_test7 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = train_size60,random_state=0) 
```

```python
dt15 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
```

```python
dt15.fit(X_train7, y_train7)
```

```python
dt15_train_acc = dt15.score(X_train7, y_train7)
dt15_train_acc
```

```python
dt15_test_acc = dt15.score(X_test7, y_test7)
dt15_test_acc
```

```python
d_60 = {'train_size':train_size60,
     'n_train_samples': X_train7.shape[0],
     'n_test_samples': X_test7.shape[0],
        'train_acc': dt15_train_acc,
        'test_acc' : dt15_test_acc}
```

```python
train_size70 = 0.7
```

```python
X_train8, X_test8, y_train8, y_test8 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = train_size70,random_state=0) 
```

```python
dt16 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
```

```python
dt16.fit(X_train8, y_train8)
```

```python
dt16_train_acc = dt16.score(X_train8, y_train8)
dt16_train_acc
```

```python
dt16_test_acc = dt16.score(X_test8, y_test8)
dt16_test_acc
```

```python
d_70 = {'train_size':train_size70,
     'n_train_samples': X_train8.shape[0],
     'n_test_samples': X_test8.shape[0],
        'train_acc': dt16_train_acc,
        'test_acc' : dt16_test_acc}
```

```python
train_size80 = 0.8
```

```python
X_train9, X_test9, y_train9, y_test9 = train_test_split (df1.values[:, :9].astype('int'), df1.values[:, 9],train_size = train_size80,random_state=0) 
```

```python
dt17 = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf = 3, criterion='entropy', min_samples_split= 4)
```

```python
dt17.fit(X_train9, y_train9)
```

```python
dt17_train_acc= dt17.score(X_train9, y_train9)
dt17_train_acc
```

```python
dt17_test_acc = dt17.score(X_test9, y_test9)
dt17_test_acc
```

```python
d_80 = {'train_size':train_size80,
     'n_train_samples': X_train9.shape[0],
     'n_test_samples': X_test9.shape[0],
        'train_acc': dt17_train_acc,
        'test_acc' : dt17_test_acc}
```

```python
#LIst of dictionaries
datasets_info_dict =[d_20, d_30, d_40,d_50, d_60, d_70,d_80]
```

```python
#making empty list for the dictionary "d"
ls_train_pct=[]
ls_n_train_samples=[]
ls_n_test_samples=[]
ls_train_acc=[]
ls_test_acc=[]
```

```python
#A loop to assign values of each dictionary to their corresponding lists
for dic in datasets_info_dict:
    train_pct = dic['train_size']
    n_train_samples = dic['n_train_samples'] 
    n_test_samples = dic['n_test_samples']
    train_acc = dic['train_acc']
    test_acc = dic['test_acc']
    
    ls_train_pct.append(train_pct)
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


## cross validation


Let's have a glance at the cross validation method. 

```python
cv_scores = cross_val_score(dt17,df1.values[:,:9],df1.values[:,9],cv=100 )
```

```python
cv_scores
```

```python
np.mean(cv_scores)
```

The resulted accuracy is close to the normal train-test splits.

```python

```
