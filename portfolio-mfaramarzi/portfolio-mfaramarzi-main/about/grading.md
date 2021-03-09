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

```python
def compute_grade(num_level1, num_level2, num_level3):
    
    if num_level1 == 15 and num_level2 == 15 and num_level3 == 15:
        return(print('letter_grade : A'))
    
    
    elif num_level1 < 15:
        
        if 10 <= num_level1 < 15:
            return(print('letter_grade : C-'))
        
        elif 5 <= num_level1 < 10:
            return(print('letter_grade : D+'))
        
        elif 0 <= num_level1 < 5:
            return(print('letter_grade : D'))
        
    elif num_level1 == 15 and num_level2 < 15:
        
        if 10 <= num_level2 < 15:
            return(print('letter_grade : B-'))
        
        elif 5 <= num_level2 < 10:
            return(print('letter_grade : C+'))
        
        elif 0 <= num_level2 < 5:
            return(print('letter_grade : C'))
    
    elif num_level1 == 15 and num_level2==15 and num_level3 < 15:  
            
        if 10 <= num_level3 < 15:
            return(print('letter_grade : A-'))
        
        elif 5 <= num_level3 < 10:
            return(print('letter_grade : B+'))
        
        elif 0 <= num_level3 < 5:
            return(print('letter_grade : B'))
            
            
            
        
```

```python
compute_grade(15, 14, 14)
```

```python
compute_grade(15, 15, 7)
```

```python
compute_grade(15, 2, 1)
```

```python

```
