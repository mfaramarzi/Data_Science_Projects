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

**Dataset #1: This dataset (d1) includes at least two continuous valued variables and one categorical variable.**

* Display overall summary statistics for a subset of 5 variables of your choice or all variables if there are fewer than 5 numerical values

* Display overall summary statistics grouped by a categorical variable

* For two continuous variables make a scatter plot and color the points by a categorical variable

* Pose one question for this dataset that can be answered with summary statistics, compute a statistic and plot that help answer that exploratory question.



# Eploratory Data Analysis (EDA) on Titanic suvivals Dataset


![alt text](titanic2.jpg)


## Importing libraries

```python
import pandas as pd # Importing pandas  for data manipulation and analysis.
import matplotlib.pyplot as plt
import seaborn as sns
```

## Reading Data

```python
#Read a csv file into DataFrame.
d1 = pd.read_csv('/home/masoud/masoud/CSC_310/week3/03-exploratory-data-analysis-mfaramarzi/data/Titanic/main_db/train.csv')
```

## Descriptive Data Analysis

```python
d1.head() #Return the first n rows.
#It is useful for quickly testing if your object has the right type of data in it.
```

```python
d1.describe() #General descriptive statistics.
```

```python
d1[["Age", "Fare"]].describe() #escriptive statistic of two specific varibales
```

<!-- #region -->
### **Types Of Features**


**Categorical Features:**

A categorical variable is one that has two or more categories and each value in that feature can be categorised by them.For example, gender is a categorical variable having two categories (male and female). Now we cannot sort or give any ordering to such variables. They are also known as Nominal Variables.

Categorical Features in the dataset: Sex,Embarked, Survived

**Ordinal Features:**

An ordinal variable is similar to categorical values, but the difference between them is that we can have relative ordering or sorting between the values. For eg: If we have a feature like Height with values Tall, Medium, Short, then Height is a ordinal variable. Here we can have a relative sort in the variable.

Ordinal Features in the dataset: PClass

**Continous Feature:**

A feature is said to be continous if it can take values between any two points or between the minimum or maximum values in the features column.

Continous Features in the dataset: Age, Fare

<!-- #endregion -->

```python
d1.dtypes #checking types of data in the dataframe
```

```python
d1.shape #Shape of Dataframe
```

```python
d1.set_index('PassengerId')# Using passenger ID as the index
```

## Display overall summary statistics grouped by a categorical variable

```python
d1.groupby('Sex').describe()
```

```python
d1.groupby("Sex").mean() # Display mean for variables chategorized by "sex"
```

## For two continuous variables make a scatter plot and color the points by a categorical variable

```python
import seaborn as sns
```

```python
d1.columns#showing columns of dataframe
```

```python
sns.relplot(x= 'Fare',y='Age',
           data=d1, hue='Sex')

#Figure-level interface for drawing relational plots onto a FacetGrid
#Using hue='Sex' we Group passengers. It will produce elements with different colors. 
```

```python
sns.relplot(x= 'Fare',y='Age',
           data=d1, hue='Survived')
#Using hue='Sex' we Group passengers based on thier gender.
```

```python
sns.relplot(x= 'Fare',y='Age',
           data=d1, hue='Embarked')
#Using hue='Sex' we Group variable based on thier embarking port.
```

## Pose one question for this dataset that can be answered with summary statistics, compute a statistic and plot that help answer that exploratory question.

**What is the average (mean) age between all passengers?**

```python
d1["Age"].mean()#the average (mean) age between all passengers
```

```python
#Plotting the correlation between passenger class and passengers age (for three different classes of passengers).

for x in [1,2,3]:    ## for 3 classes
    d1.Age[d1.Pclass == x].plot(kind="kde")
plt.title("Age wrt Pclass")
plt.legend(("1st","2nd","3rd"))
```

As we can understand from this plot lower classes are younger people on average


## Extra analysis


**grouping dataframe by one and two cathegorical data and conducting statsistical analysis on them**

```python
d1.groupby('Sex')['Survived'].count()#number of survived people for each gender
```

```python
d1.groupby(['Sex','Survived'])['Survived'].count()#Number of survived and unsurvived people for each gender
```

```python
d1.groupby('Sex')['Age'].min()#Minimum age between passengers for each gender
```

```python
d1.groupby('Sex')['Age'].max()#Maximum age between passengers for each gender
```

```python
d1.groupby('Sex')['Age'].mean()#Average age between passengers for each gender
```

```python
d1['Age'].max()#Maximum age between passengers
```

```python
d1['Age'].min() #Minimum age between passengers
```

### checking for total null values

```python
d1.isnull().sum() #Detect missing values.
```

**Boxplot plot For a continuous data**

```python
d1.boxplot('Age')#Boxplot for distribution of age between all passengers
```

When reviewing a box plot, an outlier is defined as a data point that is located outside the whiskers of the box plot. Therefore, we have outliers at the top of the plot which show the age of oldest passengers.

```python
d1.boxplot('Age', by='Sex', figsize=(12,8))#Boxplot for distribution of age for each gender
```

As we can see in the above boxplot, there are outliers in the age datafor men.All outliers are men.


We can also use matplotlib and seaborn to show the number of survided and unsurvided people in the column and pie charts 

```python
f,ax=plt.subplots(1,2,figsize=(18,8))# Number of rows/columns and fig size of the subplot grid
d1[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])#First bar plot. First plot shows the ration of survied 
ax[0].set_title('Survived vs Sex')#setting title of first plot
sns.countplot('Sex',hue='Survived',data=d1,ax=ax[1])#second plot
ax[1].set_title('Sex:Survived vs Dead')#setting title of second plot
plt.show()
```

This looks interesting. The number of men on the ship is lot more than the number of women. Still the number of women saved is almost twice the number of males saved. The survival rates for a women on the ship is around 75% while that for men in around 18-19%.


```python

f,ax=plt.subplots(1,2,figsize=(18,8))# Number of rows/columns and fig size of the subplot grid
d1['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)#First plot (pie plot)
ax[0].set_title('Survived')#title of pie plot
ax[0].set_ylabel('')#The label string of pie plot

sns.countplot('Survived',data=d1,ax=ax[1])#Show the counts of observations in each categorical bin using bars.
ax[1].set_title('Survived')#setting title of countplot
plt.show()#showing plots
```

It is evident that not many passengers survived the accident.

Out of 891 passengers in training set, only around 350 survived i.e Only 38.4% of the total training set survived the crash. We need to dig down more to get better insights from the data and see which categories of the passengers did survive and who didn't.


### Analysing correlation between survival and passenger class in a crosstab table

```python
pd.crosstab(d1.Pclass,d1.Survived,margins=True).style.background_gradient(cmap='summer_r')#Compute a simple cross tabulation of two factors
#Color the background in a gradient according to the data in each column 
```

This crosstab table confirms how passenger class did affect the survival of passengers.
class "1" passengers mostly survived, while the chance of survival was almost fifty-fifty for passengers of class "2".
Most of passengers in class "3" did not survive. 

```python

```
