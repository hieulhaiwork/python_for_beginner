# Python Review

In this repo, I will list some operations and functions that I find them popular in my learning and working.

[1. Basic operations](#1-basic-operations)
  - [a. Mathematics operations](#a-mathematics-operations)
  - [b. Function](#b-function)
  - [c. Class](#c-class)

[2. Numpy](#2-numpy)
  - [a. Array](#a-array)
  - [b. Basic operations](#b-basic-operations)
  - [c. Zeros and ones](#c-zeros-and-ones)
  - [d. Reshape](#d-reshape)
  - [e. Arange and linspace](#e-arange-and-linspace)
  - [f. Concatenate](#f-concatenate)
  - [g. Random](#g-random)
  - [h. Linalg](#h-linalg)

[3. Pandas](#3-pandas)
  - [a. DataFrame](#a-dataframe)
  - [b. Basic operations](#b-basic-operations)
  - [c. info()](#c-info)
  - [d. describe()](#d-describe)
  - [e. copy()](#e-copy)
  - [f. NULL values handling](#f-null-values-handling)
  - [g. apply()](#g-apply)

## 1. Basic operations

### a. Mathematics operations

```(python)
# Assign variables
a = 1
b = a * 5
c = b - 4

# Root and exponent
a^4 = a**4
√a = a**(1/2)
∛a = a**(1/3)
1/a = a**(-1)

# Loop for, while
sum = 0
for i in range(4):
  sum += i

i = 0
sum = 0
while i < 4:
  sum += i
  i += 1

# List (ordered and mutable)
list = [i for i in range(4)]  # get the list of intergers from 0 to 3: [0,1,2,3]
list[0]       # the first element: 0
list[-1]      # the last element: 4
list[0:2]     # sublist from first element to second element: [0,1]
list[2:]      # sublist from third element to end of list: [2,3]
list[2] = 4   # change the value of the third element to 4

# Tuple (ordered and immutable)
tuple = (1, 2) | (1, 2, 3, 1, ...)
tuple[0]      # the first element: 1
tuple[2:5]    # same as list
tuple[1] = 4  # not exist cause tuple cannot be changed after its created

# Set (unordered and mutable)
set = {1,2,3,4} # in set, every elements are unique to one another
set[0]          # not exist cause set is unordered
set.add(5)      # add new element to set
set.remove(2)   # remove 2 from set, but if 2 not in set at the beginning, error occurs
set.discard(2)  # remove 2 from set, if 2 not in set at the beginning, error not occurs
set_union = set1 | set2          # union of set1 and set2
set_intersection = set1 & set2   # intersection of set1 and set2
set_difference = set1 - set2     # difference between set1 and set2

# Dictionary
dict = {
  'key': 'value',
  1: 2,
  'hello': 0,
  'list': [1,2,5,3],
  'dict2': {
      'key': 'value'
  }
}  # followed by the rule: one key - one value, the keys in dict are unique to one another
```
### b. Function
```(python)
def function_name(parameter_1: int, parameter_2: str) -> Any:
    <what does your function do?>
    return None
```
### c. Class
#### Basic
```(python)
class class_name():
  def __init__(self):
    self.a = 1
    self.b = 0
    ....
  def do_work(self):
    return None  
```
#### Advanced
##### 1. Inheritance
```(python)
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def bit(self):
        return "Ngow"

dog = Dog()
print(dog.speak())
# Output: Some sound
```
##### 2. Class Methods and Static Methods
```(python)
# classmethod
class Animal:
  def __init__(self, name):
    self.name = name

  @classmethod
  def create(cls, name):
    return cls(name)
class Dog(Animal):
  pass
class Cat(Animal):
  pass

dog = Dog.create("Buddy")
cat = Cat.create("Meow")

print(dog.name)  # Output: Buddy
print(cat.name)  # Output: Meow

#staticmethod
class Math:
    @staticmethod
    def add(a, b):
        return a + b
result = Math.add(3, 5)
print(result)  # Output: 8
```
## 2. Numpy

Mostly used for operations with matrix and tensor.
Below are popular functions that I often use.
```(python)
import numpy as np
```
### a. Array
```(python)
matrix = np.array([1,2])  # create a matrix shape (1,2) but in python this will be shown as (2,)
matrix = np.array([[1,2,3],[2,3,4]])  # create a matrix shape (2,3)
tensor = np.array([
                    [[1,2,3], [1,2,3]],
                    [[4,5,6], [4,5,6]]
])
# matrix | tensor: np.ndarray

matrix.shape   # get shape of the matrix (rows, cols)
matrix.size    # get the total count of elements in matrix 
matrix.ndim    # get the dimensions of matrix: 2
matrix.dtype   # get dtype of matrix, some functions in other libraries require specified dtype, default: int32 for int and float32 for float values
also matrix = np.array([[1,2,3],[2,3,4]], dtype= np.int8)
```
### b. Basic operations
```(python)
# slicing
matrix_a = np.array([[1,2,3],[2,3,4]])
output = matrix_a[0:2,0:2]    # get rows from 0 to second and cols from 0 to second: array([[1,2],[2,3]])

# copy
a = matrix_a           # assign matrix_a to variable a, then change a will change matrix_a
a = matrix_a.copy() = np.copy(matrix_a)    # copy the structure and value of matrix_a to a, change a not affect to matrix_a

# add (two matrix must have the same shape)
output = matrix_a + matrix_b   # Add the values ​​in the corresponding position

# substract (two matrix must have the same shape)
output = matrix_a - matrix_b   # Substract the values ​​in the corresponding position

# multiple (two matrix must have the same shape)
output = matrix_a * matrix_b   # Multiple the values ​​in the corresponding position

# root and exponent
output = matrix_a ** 2
output = np.exp(matrix_a)

# logic
output = matrix_a < 35  # return a logic matrix with same shape as matrix_a to compare between each element in matrix and 35 then return True or False
output = matrix_a[matrix_a < 35]  # return the matrix contains the value in matrix_a satisfied that < 35

# dot product (matrix_a has shape (m,n) then matrix_b must have shape (n,z))
output = np.dot(matrix_a, matrix_b) = matrix_a @ matrix_b
```
### c. Zeros and ones
```(python)
zero = np.zeros((3,4), dtype=np.int8)         # create a matrix shape (3,4) with all value 0
ones = np.ones((3,4,2))                       # create a tensor shape (3,4,2) with all value 1
```
### d. Reshape
```(python)
matrix = np.array([[1,2,3],[2,3,4]])      # shape: (2,3)
reshaped_matrix = matrix.reshape((3,2))   # shape: (3,2)
Notice that: reshaped_matrix.size must be equal to matrix.size or error will occur
```
### e. Arange and linspace
```(python)
a = np.arange(0,50,10)    # this function get start, end and step like function range but return output as array
# Ouptut: array([0,10,20,30,40])

b = np.linspace(0,50,num = 10)  # similar to arange but using number of samples instead of step
# Output: array([ 0.        ,  5.55555556, 11.11111111, 16.66666667, 22.22222222, 27.77777778, 33.33333333, 38.88888889, 44.44444444, 50.        ])
```
### f. Concatenate
```(python)
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([7,8])
result = np.concatenate((a, b), axis=0)  # Concat 2 matrix by axis (ordered by your choice but common order: 0 (rows), 1 (cols), 2 (channels in image), ...)
# Output: array([1, 2, 3, 4, 5, 6])
```
### g. Random
Below are popular syntax to get random set but class random in numpy can do more than that including get random values follows by uniform distribution, rayleigh distribution,.... 
```(python)
output = np.random.randint(10)        # Get random 1 interger value from 0 to 10
output = np.random.randint(10,50)     # Get random 1 interger value from 10 to 50
output = np.random.randint(100, size=(10))  # Get random 10 interger values from 0 to 100 and return output as np.ndarray
output = np.random.rand(10)           # Get random 10 float values from 0 to 1
output = np.random.rand(3, 5)         # Get 2-D array has 3 rows, each row has 5 values
output = np.random.choice([1,2,3])    # Get 1 random value from an array (np.ndarray, list or tuple, not set or dict)
output = np.random.choice([1,2,3], size=4)      # Return a numpy array with 4 random values in [1,2,3]
output = np.random.choice([1,2,3], size=(3,5))  # Return a numpy array with 3 rows and each row has 5 random values
```
### h. Linalg
This is powerfull tool to work with matrix, specially 2-D matrix, including norm, trace, svd,...  
Read more at: [here](https://numpy.org/devdocs/reference/routines.linalg.html)

## 3. Pandas

Pandas is a powerful library designed for handling with tabular in many formats: .csv, .xlsx, .json, .html, .sql,... But .csv is the most popular format file to store table-format data, especially in data analysis.

```
import pandas as pd
```
### a. DataFrame
Read the csv file, the result will be shown as a table with rows and columns:
```
dataframe = pd.read_csv('path-to-your-csv')
```
or you can create your own dataframe from dictionary:
```
dict = {
  'A': [0,1,2,3]
  'B': [1990,2000,2010,2020]
  'C': ['a', 'b', 'c', 5]
  'D': [4,1,2,'one']
}

my_dataframe = pd.DataFrame(dict)
```
then when saving your dataframe to a .csv file:
```
dataframe.to_csv("path-to-new-csv")
```
### b. Basic operations
- **head and tail**: Show the content of variable `dataframe` by row
```
dataframe.head()    # Show first 5 rows of the dataframe (default)
dataframe.head(10)  # Show first 10 rows of the dataframe
dataframe.tail()    # Show last 5 rows of the dataframe (default)
dataframe.tail(10)  # Show last 10 rows of the dataframe
```
- **columns**: Return names of all columns in dataframe
```
columns = dataframe.columns
```
- **loc and iloc**: Choose and search data in dataframe
```
# Choose a single column
a = dataframe['Date']

# Choose multi columns
a = dataframe[['A', 'B', 'C']]

# .loc[index] used for searching by rows when index is index of the row
row_1 = dataframe.loc[0]
row_2_10 = dataframe.loc[1:9]  # Get rows with all columns from second row to 10th row

# .iloc[row, col] used for searching by both rows and cols' indexes.
output = dataframe.iloc[2,3]      # Get the value in row 3 and col 4
output = dataframe.iloc[:5, 2:4]  # Get sub dataframe has rows from 1st to 6th and cols from 3rd to 5th
```
- **Search by condition**:
```
a = dataframe[dataframe['Name'] == 'a']  # Return the row where Name = 'a'
```
- **Add new column**:
```
dataframe['New_column'] = [a,b,c,...]
```
- **Drop column**:
```
dataframe.drop('column_name', axis=1)
```
- **Rename column**:
```
dataframe.rename(columns={'old_name': 'new_name'})
```
- **Get mean, median and std**:
Notice that these only work on numerical columns
```
mean = dataframe['column_name'].mean()
median = dataframe['column_name'].median()
std = dataframe['column_name'].std()
```
### c. info(): 
Return the general information of dataframe including class, number of rows, number of columns, names of columns, non-null count, dtypes of each columns,...  
In data analysis, this helps to take a quick understanding of dataframe.
```
dataframe.info()
```
### d. describe():
Return statistical information of the dataframe including count, mean, std, min, max,... but only works on numerical columns
```
dataframe.describe()
```
### e. copy():
Copy dataframe to new variable:
```
new_dataframe = dataframe.copy(deep = True)  # Default: deep = True, copy all structure and data values from dataframe to new dataframe, then changing new_dataframe not affect to orinal dataframe. Otherwise if deep = False, it will only copy structure but share data values.
```
### f. Null values handling
- **isnull()** or **isna()**:
Both function share the same effect to count the null values in dataframe
```
dataframe.isnull()          # Return the null values count of all columns
dataframe['Name'].isnull()  # Return the null values count of column 'Name'
```
- **dropna()**: Drop row with null values
```
dataframe.dropna()
```
- **fillna()**: Replace null values with specified values
```
dataframe.fillna(0)  # Replace null values with 0
```
### g. apply()
This takes a function and apply it to all values in pandas series. 
```
def compare(num):
  if num < 200:
    return False
  else:
    return True
dataframe.apply(compare)
```

