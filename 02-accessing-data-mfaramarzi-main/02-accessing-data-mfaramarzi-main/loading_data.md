---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: csc310
  language: python
  name: csc310
---

# Accesing Data

+++

Find 3 datasets of interest to you that are provided in different file formats. Choose datasets that are not too big, so that they do not take more than a few second to load. At least one dataset, must have non numerical (eg string or boolean) data in at least 1 column. Complete a dictionary for each with the url, a name, and what function should be used to load the data into a pandas.DataFrame.

```{code-cell} ipython3
# imports
import pandas as pd
```

```{code-cell} ipython3
# Three disctionaries for three websites

d1 = {'url':'https://www.numbeo.com/quality-of-life/rankings_by_country.jsp',
     'name': 'rankings_by_country',
     'load_func': lambda URL: pd.read_html(URL)[1]}
                                   
#what function should be used to load the data into a pandas.DataFrame
#.what should be input of func?url?
    
d2 = {'url': 'https://en.wikipedia.org/wiki/List_of_best-selling_music_artists' ,
     'name': 'List_of_best-selling_music_artists' ,
     'load_func': lambda URL: pd.read_html(URL)[0]}

d3 = {'url': 'https://finance.yahoo.com/quote/TSLA/history' ,
     'name': 'history',
     'load_func': lambda URL: pd.read_html(URL)[0]}
#d3['url']
```

```{code-cell} ipython3
# Making a list of dictionaries

datasets_info_dict =[d1, d2, d3]
```

## Dataset summary


```{code-cell} ipython3
website_1 = pd.read_html(d1['url'])
```

```{code-cell} ipython3
# df1 is the second index of website_1 list
df1 = d1['load_func'](d1['url'])
```

```{code-cell} ipython3
type(df1)
```

```{code-cell} ipython3
df1.shape
```

```{code-cell} ipython3
# Showing the first five components of data frame (df1)

df1.head()
```

```{code-cell} ipython3
# Showing the last five components of data frame (df1)

df1.tail()
```

### Display the heading with the last seven rows

```{code-cell} ipython3
df1.tail(7)
```

### make and display a new data frame with only the non numerical columns

```{code-cell} ipython3
df1.select_dtypes(exclude='number').style.hide_index() 
```

```{code-cell} ipython3
df1.index
```

```{code-cell} ipython3
df1.describe()
```

### Was the format that the data was provided in a good format? why or why not?

+++

Yes. Because it was provided in rows and columns and data type was consistent in each column. 

```{code-cell} ipython3
website_2 = pd.read_html(d2['url'])
```

```{code-cell} ipython3
df2 = d2['load_func'](d2['url'])
df2
```

### Display the heading and the first three rows

```{code-cell} ipython3
df2.head(3)
```

### display the datatype for each column

```{code-cell} ipython3
df2.dtypes
```

### Are there any variables where pandas may have read in the data as a datatype that’s not what you expect (eg a numerical column mistaken for strings)?

+++

no. all columns include non numerical characters.

```{code-cell} ipython3
website_3= pd.read_html(d3['url'])
```

```{code-cell} ipython3
df3 = d3['load_func'](d3['url'])
df3
```

```{code-cell} ipython3
df3['Date']
```

## Display the first 5 even rows of the data for three columns of your choice

```{code-cell} ipython3
df3[['Date','Low', 'Volume']].loc[::2].head()
# df3.Date.Low.Volume.loc[::2].head() #Does not work!
```

### Try reading it in with the wrong read_ function. If you had done this by accident, how could you tell?

```{code-cell} ipython3
website_3 = pd.read_csv('https://finance.yahoo.com/quote/TSLA/history')
```

I can undesrtand it by the error I am getting, as it says this fuction is not able to parse data

```{code-cell} ipython3

type(d1)
```

 Use a list of those dictionaries to iterate over the datasets and build a table that describes them, with the following columns ['name','source','num_rows', 'num_columns','source_file_name']. The source column should be the url where you loaded the data from or the source if you downloaded it from a website first The source_file_name should be the part of the url after the last /, you should extract this programmatically. Display that summary table as a dataframe and save it as a csv, named dataset_summary.csv.

```{code-cell} ipython3
a = []
b = []
c = []
d = []
e = []


for dic in datasets_info_dict:
    name_website = dic['name']
    address_website = dic['url'] 
    num_rows_df = dic['load_func'](dic['url'])[0]
    num_columns_df = dic['load_func'](dic['url'])[1]
#     if i==1:
#         num_rows_df = df1.shape[0]
#         num_columns_df = df1.shape[1]
#     elif i==2:
#         num_rows_df  = df2.shape[0]
#         num_columns_df = df2.shape[1]
#     elif i==3:
#         num_rows_df  = df3.shape[0]
#         num_columns_df = df3.shape[1]
    
    source_file_name_website = dic['url'].split('/')[-1]
    
    a.append(name_website)
    b.append(address_website)
    c.append(num_rows_df)
    d.append(num_columns_df)
    e.append(source_file_name_website)
    
    ex_dic = {'name' : a,
             'source' : b,
             'num_rows' : c,
             'num_columns' : d,
              'source_file_name' : e,
             }
columns = ['name', 'source', 'num_rows', 'num_columns', 'source_file_name']
index = ["d_1", "d_2", "d_3"]
df = pd.DataFrame(ex_dic, columns=columns, index=index)
df
df.to_csv('dataset_summary.csv') 
#print(ex_dic)
```

## (for the process skill)

### Make a list of a data science pipeline and denote which types of programming might be helpful at each staged. Include this in a markdown cell in the same notebook with your analysis'''

+++

    1. O — Obtaining our data (MySQL)
    2. S — Scrubbing / Cleaning our data (Python or R)
    3. E — Exploring / Visualizing our data will allow us to find patterns and trends (Numpy, Matplotlib, Pandas or Scipy)
    4. M — Modeling our data will give us our predictive power as a wizard (Sci-kit Learn)
    5. N — Interpreting our data
    
[reference] (https://towardsdatascience.com/a-beginners-guide-to-the-data-science-pipeline-a4904b2d8ad3)

```{code-cell} ipython3

```
