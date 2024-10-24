# Python Review

# 1. Basic operations

## a. Mathematics operations

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
# 2. Numpy

Mostly used for operations with matrix and tensor.
Below are popular functions that I often use.
```(python)
import numpy as np
```
## a. array
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
## b. basic operations
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
## c. zeros and ones
```(python)
zero = np.zeros((3,4), dtype=np.int8)         # create a matrix shape (3,4) with all value 0
ones = np.ones((3,4,2))                       # create a tensor shape (3,4,2) with all value 1
```
## d. reshape
```(python)
matrix = np.array([[1,2,3],[2,3,4]])      # shape: (2,3)
reshaped_matrix = matrix.reshape((3,2))   # shape: (3,2)
Notice that: reshaped_matrix.size must be equal to matrix.size or error will occur
```
## e. arange and linspace
```(python)
a = np.arange(0,50,10)    # this function get start, end and step like function range but return output as array
# Ouptut: array([0,10,20,30,40])

b = np.linspace(0,50,num = 10)  # similar to arange but using number of samples instead of step
# Output: array([ 0.        ,  5.55555556, 11.11111111, 16.66666667, 22.22222222, 27.77777778, 33.33333333, 38.88888889, 44.44444444, 50.        ])
```
## f. concatenate
```(python)
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([7,8])
result = np.concatenate((a, b), axis=0)  # Concat 2 matrix by axis (ordered by your choice but common order: 0 (rows), 1 (cols), 2 (channels in image), ...)
# Output: array([1, 2, 3, 4, 5, 6])
```
## g. random
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
## h. linalg
This is powerfull tool to work with matrix, specially 2-D matrix, including norm, trace, svd,...  
Read more at: [here](https://numpy.org/devdocs/reference/routines.linalg.html)

# 3. Pandas

Pandas is a powerful library designed for handling with tabular in many formats: .csv, .xlsx, .json, .html, .sql,... But .csv is the most popular format file to store data, especially in data analysis.

```
import pandas as pd
```
## a. read_csv
Read the content in a csv file, the result will be saved as table-form with rows and columns
```
dataframe = pd.read_csv('path-to-your-csv')
```
## b. head and tail
Show the content of variable `dataframe` by row:
```
dataframe.head()    # Show first 5 rows of the dataframe (default)
dataframe.head(10)  # Show first 10 rows of the dataframe
dataframe.tail()    # Show last 5 rows of the dataframe (default)
dataframe.tail(10)  # Show last 10 rows of the dataframe
```
## c. sclicing

# 4. Matplotlib and seaborn

# 5. Scikit-learn

# 6. Opencv
## a. merge

# 7. nltk

