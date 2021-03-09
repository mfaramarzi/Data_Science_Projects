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

### For d2:

* Display two individual summary statistics for one variable

* Group the data by two categorical variables and display a table of one summary statistic

* Use a seaborn plotting function with the col parameter or a FacetGrid to make a plot that shows something informative about this data, using both categorical variables and at least one numerical value. Describe what this tells you about the data.

* Produce one additional plot of a different plot type that shows something about this data.


# Eploratory Data Analysis (EDA) on FIFA 19 complete player dataset

**18k+ FIFA 19 players, ~90 attributes extracted from the latest FIFA database**


![alt text](FIFA.jpeg)


## Importing libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

## Reading Data

```python
#Read a csv file into DataFrame.
df = pd.read_csv('/home/masoud/masoud/CSC_310/week3/03-exploratory-data-analysis-mfaramarzi/data/FIFA 19 complete player dataset/data.csv')
```

## Descriptive Data Analysis

```python
df.head() #Return the first n rows of raw data.
#It is useful for quickly testing if your object has the right type of data in it.
```

```python
df.describe() #General descriptive statistics.
```

```python
df.describe().T #We can switch raws and columns
```

```python
df.info() #we can have general information about our data as below
```

```python
df.shape #Dataframe shape

```

```python
df.shape[0] #Dataframe number of rows
```

```python
df.shape[1] #Dataframe number of columns
```

```python
df.head().T #We can swith Rows and columns
```

## Display two individual summary statistics for one variable

```python
df['Age'].describe() #Describing statistics for the age of players
```

```python
df['Age'].max() #Maximum age between players
```

```python
df['Age'].min() #Minimum age between players
```

```python
df['Age'].mean() #Average age of players
```

```python
df.columns #Listing name of columns in the dataframe
```

## Group the data by two categorical variables and display a table of one summary statistic

```python
df.groupby(['Body Type','Position']).describe().head(50)#Grouping data by two atributes of 'Body Type' and 'Position' and showing table of one summary statistic for the first 50 inputs
```

```python
df.groupby(['Nationality','Position']).describe().tail(50) #Grouping data by two atributes of 'Nationality' and 'Position' and showing table of one summary statistic for the last 50 inputs
```

## Use a seaborn plotting function with the col parameter or a FacetGrid to make a plot that shows something informative about this data, using both categorical variables and at least one numerical value. Describe what this tells you about the data.


**We can see wage vs age for different player positions using seaborn.replot and choosing col="Position"**

```python
sns.relplot( 
    data=df, x="Age", y="Wage", col="Position", kind="line",
) #It is a Figure-level interface for drawing relational plots onto a FacetGrid.
```

## Produce one additional plot of a different plot type that shows something about this data.


**Using seaborn we can show the correlation between 'Potential' and 'Wage' of players considering their positioin**

```python
sns.relplot(x= 'Potential',y='Wage',data=df, hue='Position').fig.set_size_inches(15,15)
```

**In the three below plots, using seaborn displot we can show the distribution of number of players at different ages, distinguished with their positions**

```python
sns.displot(data=df, x="Age", hue="Position") # with kind="hist"; the default
```

```python
sns.displot(data=df, x="Age", hue="Position", kind="kde") # with kind="kde"
```

**Using seaborn.jointplot we can draw a plot of two variables with bivariate and univariate graphs**

```python
sns.jointplot(x=df['Age'],y=df['Potential'],
              joint_kws={'alpha':0.1,'s':5,'color':'red'},
              marginal_kws={'color':'yellow'}) #by default kind is 'scatter' as it is here
```

 In the above plot we show potential tends to fall as players grow old.

```python
sns.jointplot(x=df['Age'],y=df['Potential'],
              joint_kws={'alpha':0.1,'s':5,'color':'blue'},
              marginal_kws={'color':'black'}, kind="kde") #Setting kind="kde" will draw both bivariate and univariate KDEs:
```

```python

```
