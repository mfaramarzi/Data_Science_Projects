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
    display_name: Python 3
    language: python
    name: python3
---

# Naive Bayes Classification

For each dataset, answer the following:

1. Do you expect Gaussian Naive Bayes to work well on this dataset, why or why not? 
    - think about the assumptions of naive bayes and classification in general) 
    - _explanation is essential here, because you can actually use the classifier to check_

1. How well does a Gaussian Naive Bayes classifier work on this dataset? Do you think a different classifier might work better or do you think this data cannot be predicted any better than this?
    - check both the overall performance and the type of errors
    - are the errors random or are some errors more common than others

1. How does the actual performance compare to your prediction?  If it performs much better or much worse than you expected, what might you use to figure out why? 
> _you do not have to figure out why your predictions were not correct, just list tools you've learned in class that might help you figure that out_

```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
```

# Dataset 1

```python
df1 = pd.read_csv('data/dataset1.csv')
sns.pairplot(data =df1, hue='char')
```

```python

```

```python

```

```python

```

# Dataset 2

```python
df2 = pd.read_csv('data/dataset2.csv')
sns.pairplot(data =df2, hue='char')
```

```python

```

```python

```

```python

```

# Dataset 3

```python
df3 = pd.read_csv('data/dataset3.csv')
sns.pairplot(data =df3, hue='char')
```

```python

```

```python

```

```python

```

# Dataset 4

```python
df4 = pd.read_csv('data/dataset4.csv')
sns.pairplot(data =df4, hue='char')
```

```python

```

```python

```

```python

```

# Dataset 5

```python
df5 = pd.read_csv('data/dataset5.csv')
sns.pairplot(data =df5, hue='char')
```

```python

```

```python

```

```python

```

# Dataset 6

```python
df6 = pd.read_csv('data/dataset6.csv')
sns.pairplot(data =df6, hue='char')
```

```python

```

```python

```

```python

```
