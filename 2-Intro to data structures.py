# https://pandas.pydata.org/docs/user_guide/dsintro.html
##############################################################
# Intro to data structures
##############################################################

# We’ll start with a quick, non-comprehensive overview of the fundamental data structures
# in pandas to get you started. The fundamental behavior about data types, indexing,
# axis labeling, and alignment apply across all of the objects.
# To get started, import NumPy and load pandas into your namespace:
import numpy as np
import pandas as pd

# Fundamentally, data alignment is intrinsic. The link between labels and
# data will not be broken unless done so explicitly by you.
#
# We’ll give a brief intro to the data structures, then consider
# all of the broad categories of functionality and methods in separate sections.

##############################################################
# Series
##############################################################

# Series is a one-dimensional labeled array capable of holding any data
# type (integers, strings, floating point numbers, Python objects, etc.).
# The axis labels are collectively referred to as the index. The basic
# method to create a Series is to call:
# s = pd.Series(data, index=index)
# Here, data can be many different things:
# a Python dict
# an ndarray
# a scalar value (like 5)
# The passed index is a list of axis labels. Thus, this separates into a few
# cases depending on what data is:
# From ndarray
# If data is an ndarray, index must be the same length as data.
# If no index is passed, one will be created having
# values [0, ..., len(data) - 1].
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print(s)
print(s.index)
print(pd.Series(np.random.randn(5)))

# pandas supports non-unique index values. If an operation that does not
# support duplicate index values is attempted, an exception will be raised
# at that time.

# From dict
# Series can be instantiated from dicts:
d = {"b": 1, "a": 0, "c": 2}
print(pd.Series(d))
print(pd.Series(d, index=["b", "c", "d", "a"]))

# NaN (not a number) is the standard missing data marker used in pandas.

# From scalar value
# If data is a scalar value, an index must be provided.
# The value will be repeated to match the length of index.
print(pd.Series(5.0, index=["a", "b", "c", "d", "e"]))
print(pd.Series("asd", index=["a", "b", "c", "d", "e"]))

##############################################################
# Series is ndarray-like
##############################################################

# Series acts very similarly to a ndarray and is a valid argument to most NumPy functions.
# However, operations such as slicing will also slice the index.
print(s)
print(s.iloc[0])
print(s.iloc[:3])
print(s[s > s.median()])
print(s.iloc[[4, 3, 1]])
print(np.exp(s))

# We will address array-based indexing like s.iloc[[4, 3, 1]] in section on indexing.

# Like a NumPy array, a pandas Series has a single dtype.
print(s.dtype)

# This is often a NumPy dtype. However, pandas and 3rd-party libraries extend NumPy’s type
# system in a few places, in which case the dtype would be an ExtensionDtype.
# Some examples within pandas are Categorical data and Nullable integer data type. See dtypes for more.
# If you need the actual array backing a Series, use Series.array.
print(s.array)

# Accessing the array can be useful when you need to do some operation without the index
# (to disable automatic alignment, for example).

# Series.array will always be an ExtensionArray. Briefly, an ExtensionArray is a thin wrapper
# around one or more concrete arrays like a numpy.ndarray. pandas knows how to take
# an ExtensionArray and store it in a Series or a column of a DataFrame. See dtypes for more.

# While Series is ndarray-like, if you need an actual ndarray, then use Series.to_numpy().
print(s.to_numpy())

##############################################################
# Series is dict-like
##############################################################

# A Series is also like a fixed-size dict in that you can get and set values by index label:
print(s["a"])
print(s["e"])
print(s)
print("e" in s)
print("f" in s)

# If a label is not contained in the index, an exception is raised:
# UNCOMMENT TO SEE THE EXCEPTION
# print(s["f"])

# Using the Series.get() method, a missing label will return None or specified default:
print(s.get("f"))
print(s.get("f", np.nan))
# These labels can also be accessed by attribute.

##############################################################
# Vectorized operations and label alignment with Series
##############################################################

# When working with raw NumPy arrays, looping through value-by-value is usually not necessary.
# The same is true when working with Series in pandas. Series can also be passed into most
# NumPy methods expecting an ndarray.

print(s)
print(s + s)
print(s * 2)
print((np.exp(s)))

# A key difference between Series and ndarray is that operations between Series automatically
# align the data based on label. Thus, you can write computations without giving consideration
# to whether the Series involved have the same labels.

print(s.iloc[1:])
print(s.iloc[:-1])
print(s.iloc[1:] + s.iloc[:-1])

# The result of an operation between unaligned Series will have the union of the indexes involved.
# If a label is not found in one Series or the other, the result will be marked as missing NaN.
# Being able to write code without doing any explicit data alignment grants immense freedom and
# flexibility in interactive data analysis and research. The integrated data alignment features
# of the pandas data structures set pandas apart from the majority of related tools for working
# with labeled data.

# In general, we chose to make the default result of operations between differently indexed objects
# yield the union of the indexes in order to avoid loss of information. Having an index label,
# though the data is missing, is typically important information as part of a computation.
# You of course have the option of dropping labels with missing data via the dropna function.

##############################################################
# Name attribute
##############################################################

# Series also has a name attribute:
s = pd.Series(np.random.randn(5), name="something")
print(s)
print(s.name)

# The Series name can be assigned automatically in many cases, in particular, when selecting
# a single column from a DataFrame, the name will be assigned the column label.

# You can rename a Series with the pandas.Series.rename() method.

s2 = s.rename("different")
print(s.name)
print(s2.name)

# Note that s and s2 refer to different objects.

##############################################################
# DataFrame
##############################################################

# DataFrame is a 2-dimensional labeled data structure with columns of potentially different types.
# You can think of it like a spreadsheet or SQL table, or a dict of Series objects.
# It is generally the most commonly used pandas object.
# Like Series, DataFrame accepts many different kinds of input:

# Dict of 1D ndarrays, lists, dicts, or Series
# 2-D numpy.ndarray
# Structured or record ndarray
# A Series
# Another DataFrame

# Along with the data, you can optionally pass index (row labels) and columns (column labels) arguments.
# If you pass an index and / or columns, you are guaranteeing the index and / or
# columns of the resulting DataFrame. Thus, a dict of Series plus a specific index will
# discard all data not matching up to the passed index.

# If axis labels are not passed, they will be constructed from the input data based on common sense rules.

##############################################################
# From dict of Series or dicts
##############################################################

# The resulting index will be the union of the indexes of the various Series. If there are any nested dicts,
# these will first be converted to Series. If no columns are passed, the columns will be
# the ordered list of dict keys.

d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}
print(d)
df = pd.DataFrame(d)
print(df)
print(pd.DataFrame(d, index=["d", "b", "a"]))
print(pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"]))

# The row and column labels can be accessed respectively by accessing the index and columns attributes:

# When a particular set of columns is passed along with a dict of data,
# the passed columns override the keys in the dict.

print(df.index)
print(df.columns)

##############################################################
# From dict of ndarrays / lists
##############################################################

# All ndarrays must share the same length. If an index is passed, it must also be the same length as the arrays.
# If no index is passed, the result will be range(n), where n is the array length.

d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
print(pd.DataFrame(d))

print(pd.DataFrame(d, index=["a", "b", "c", "d"]))

##############################################################
# From structured or record array
##############################################################

# This case is handled identically to a dict of arrays.

data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "S10")])
print(data)
print(data.shape)
data[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]
print(data)
print(data.shape)
print(pd.DataFrame(data))
print(pd.DataFrame(data, index=["first", "second"]))
print(pd.DataFrame(data, columns=["C", "A", "B"]))

# DataFrame is not intended to work exactly like a 2-dimensional NumPy ndarray.

##############################################################
# From a list of dicts
##############################################################

data2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
print(pd.DataFrame(data2))
print(pd.DataFrame(data2, index=["first", "second"]))
print(pd.DataFrame(data2, columns=["a", "b"]))

##############################################################
# From a dict of tuples
##############################################################

# You can automatically create a MultiIndexed frame by passing a tuples dictionary.
dff = pd.DataFrame(
    {
        ("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
        ("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
        ("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
        ("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
        ("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
    }
)
print(dff)
print(dff.index)
print(dff.columns)

##############################################################
# From a Series
##############################################################

# The result will be a DataFrame with the same index as the input Series, and with one column
# whose name is the original name of the Series (only if no other column name provided).

ser = pd.Series(range(3), index=list("abc"), name="ser")
dff = pd.DataFrame(ser)
print(dff)

##############################################################
# From a list of namedtuples
##############################################################

# The field names of the first namedtuple in the list determine the columns of the DataFrame.
# The remaining namedtuples (or tuples) are simply unpacked and their values are fed into the
# rows of the DataFrame. If any of those tuples is shorter than the first namedtuple then the
# later columns in the corresponding row are marked as missing values. If any are longer than
# the first namedtuple, a ValueError is raised.

from collections import namedtuple
Point = namedtuple("Point", "x y")
print(pd.DataFrame([Point(0, 0), Point(0, 3), (2, 3)]))
Point3D = namedtuple("Point3D", "x y z")
print(pd.DataFrame([Point3D(0, 0, 0), Point3D(0, 3, 5), Point(2, 3)]))

##############################################################
# From a list of dataclasses
##############################################################

# Data Classes as introduced in PEP557, can be passed into the DataFrame constructor. Passing
# a list of dataclasses is equivalent to passing a list of dictionaries.
#
# Please be aware, that all values in the list should be dataclasses, mixing types in the list
# would result in a TypeError.

from dataclasses import make_dataclass
Point = make_dataclass("Point", [("x", int), ("y", int)])
print(pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)]))

# Missing data
#
# To construct a DataFrame with missing data, we use np.nan to represent missing values. Alternatively,
# you may pass a numpy.MaskedArray as the data argument to the DataFrame constructor, and its masked
# entries will be considered missing. See Missing data for more.

##############################################################
# Alternate constructors
##############################################################

# DataFrame.from_dict

# DataFrame.from_dict() takes a dict of dicts or a dict of array-like sequences and returns a DataFrame.
# It operates like the DataFrame constructor except for the orient parameter which is 'columns' by default,
# but which can be set to 'index' in order to use the dict keys as row labels.

print(pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])])))

# If you pass orient='index', the keys will be the row labels.
# In this case, you can also pass the desired column names:

dff = pd.DataFrame.from_dict(
    dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]),
    orient="index",
    columns=["one", "two", "three"],
)
print(dff)

# DataFrame.from_records

# DataFrame.from_records() takes a list of tuples or an ndarray with structured dtype. It works analogously
# to the normal DataFrame constructor, except that the resulting DataFrame index may be a specific field of
# the structured dtype.

print(data)
print(pd.DataFrame.from_records(data, index="C"))

##############################################################
# Column selection, addition, deletion
##############################################################

# You can treat a DataFrame semantically like a dict of like-indexed Series objects. Getting, setting, and
# deleting columns works with the same syntax as the analogous dict operations:

print(df)
print(df["one"])

df["three"] = df["one"] * df["two"]
print(df)

df["flag"] = df["one"] > 2
print(df)

# Columns can be deleted or popped like with a dict:

del df["two"]
print(df)

three = df.pop("three")
print(df)

# When inserting a scalar value, it will naturally be propagated to fill the column:

df["foo"] = "bar"
print(df)

# When inserting a Series that does not have the same index as the DataFrame,
# it will be conformed to the DataFrame’s index:

df["one_trunc"] = df["one"][:2]
print(df)

# You can insert raw ndarrays but their length must match the length of the DataFrame’s index.

# By default, columns get inserted at the end. DataFrame.insert() inserts at a particular location in the columns:

df.insert(1, "bar", df["one"])
print(df)

##############################################################
# Assigning new columns in method chains
##############################################################

# Inspired by dplyr’s mutate verb, DataFrame has an assign() method that allows you to easily create
# new columns that are potentially derived from existing columns.

""" 
#in order to load iris data:
#conda install scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()
# Create DataFrame from features
df_iris = pd.DataFrame(data=iris.data, columns=["SepalLength","SepalWidth","PetalLength","PetalWidth"])
# Add the target (species) column
# Map numerical targets to species names for better readability
df_iris['species'] = iris.target
df_iris['name'] = df_iris['species'].apply(lambda x: iris.target_names[x])
del df_iris['species']
df_iris.to_csv('2-Intro to data structures/iris.data', index=False)
"""

iris = pd.read_csv("2-Intro to data structures/iris.data")
print(iris.head())

print(iris.assign(sepal_ratio=iris["SepalWidth"] / iris["SepalLength"]).head())

# In the example above, we inserted a precomputed value. We can also pass in a function of
# one argument to be evaluated on the DataFrame being assigned to.

print(iris.assign(sepal_ratio=lambda x: (x["SepalWidth"] / x["SepalLength"])).head())

# assign() always returns a copy of the data, leaving the original DataFrame untouched.

# Passing a callable, as opposed to an actual value to be inserted, is useful when you don’t have
# a reference to the DataFrame at hand. This is common when using assign() in a chain of operations.
# For example, we can limit the DataFrame to just those observations with a Sepal Length greater than 5,
# calculate the ratio, and plot:

import matplotlib.pyplot as plt
plt.close("all")

(
    iris.query("SepalLength > 5")
    .assign(
        SepalRatio=lambda x: x.SepalWidth / x.SepalLength,
        PetalRatio=lambda x: x.PetalWidth / x.PetalLength,
    )
    .plot(kind="scatter", x="SepalRatio", y="PetalRatio")
)
#plt.show()

# Since a function is passed in, the function is computed on the DataFrame being assigned to.
# Importantly, this is the DataFrame that’s been filtered to those rows with sepal length greater
# than 5. The filtering happens first, and then the ratio calculations. This is an example
# where we didn’t have a reference to the filtered DataFrame available.

# The function signature for assign() is simply **kwargs. The keys are the column names for the
# new fields, and the values are either a value to be inserted (for example, a Series or NumPy array),
# or a function of one argument to be called on the DataFrame. A copy of the original DataFrame
# is returned, with the new values inserted.

# The order of **kwargs is preserved. This allows for dependent assignment, where an expression
# later in **kwargs can refer to a column created earlier in the same assign().

dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print(dfa)
print(dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"]))

# In the second expression, x['C'] will refer to the newly created column,
# that’s equal to dfa['A'] + dfa['B'].

##############################################################
# Indexing / selection
##############################################################

# The basics of indexing are as follows:

# Operation                       Syntax              Result
# Select column                   df[col]             Series
# Select row by label             df.loc[label]       Series
# Select row by integer location  df.iloc[loc]        Series
# Slice rows                      df[5:10]            DataFrame
# Select rows by boolean vector   df[bool_vec]        DataFrame

# Row selection, for example, returns a Series whose index is the columns of the DataFrame:
print(type(df.loc["b"]))
print(df.loc["b"])

print(type(df.iloc[2]))
print(df.iloc[2])

# For a more exhaustive treatment of sophisticated label-based indexing and slicing,
# see the section on indexing. We will address the fundamentals of reindexing / conforming
# to new sets of labels in the section on reindexing.

##############################################################
# Data alignment and arithmetic
##############################################################

# Data alignment between DataFrame objects automatically align on both the columns and the index
# (row labels). Again, the resulting object will have the union of the column and row labels.

df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=["A", "B", "C"])
print(df)
print(df2)
print(df+df2)

# When doing an operation between DataFrame and Series, the default behavior is to align
# the Series index on the DataFrame columns, thus broadcasting row-wise. For example:

print(df - df.iloc[0])

# For explicit control over the matching and broadcasting behavior, see the section
# on flexible binary operations.

# Arithmetic operations with scalars operate element-wise:

print(df)
print(df * 5 + 2)
print(1 / df)
print(df ** 4)

# Boolean operators operate element-wise as well:

df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print(df1)
print(df2)
print(df1 & df2)
print(df1 | df2)
print(df1 ^ df2)
print(-df1)

##############################################################
# Transposing
##############################################################

# To transpose, access the T attribute or DataFrame.transpose(), similar to an ndarray:
# only show the first 5 rows
print(df[:5].T)

##############################################################
# DataFrame interoperability with NumPy functions
##############################################################

# Most NumPy functions can be called directly on Series and DataFrame.

print(np.exp(df))
print(np.asarray(df))

# DataFrame is not intended to be a drop-in replacement for ndarray as its indexing semantics and data model are quite different in places from an n-dimensional array.
#
# Series implements __array_ufunc__, which allows it to work with NumPy’s universal functions.
#
# The ufunc is applied to the underlying array in a Series.

ser = pd.Series([1, 2, 3, 4])
print(np.exp(ser))

# When multiple Series are passed to a ufunc, they are aligned before performing the operation.
#
# Like other parts of the library, pandas will automatically align labeled inputs
# as part of a ufunc with multiple inputs. For example, using numpy.remainder() on
# two Series with differently ordered labels will align before the operation.

ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])

print(ser1)
print(ser2)

print(np.remainder(ser1, ser2))

# As usual, the union of the two indices is taken, and non-overlapping values are filled
# with missing values.

ser3 = pd.Series([2, 4, 6], index=["b", "c", "d"])
print(ser3)
print(np.remainder(ser1, ser3))

# When a binary ufunc is applied to a Series and Index, the Series implementation takes
# precedence and a Series is returned.

ser = pd.Series([1, 2, 3])
idx = pd.Index([4, 5, 6])

print(np.maximum(ser, idx))

# NumPy ufuncs are safe to apply to Series backed by non-ndarray arrays, for example arrays.
# SparseArray (see Sparse calculation). If possible, the ufunc is applied without converting
# the underlying data to an ndarray.

##############################################################
# Console display
##############################################################

# A very large DataFrame will be truncated to display them in the console. You can also get
# a summary using info(). (The baseball dataset is from the plyr R package):

baseball = pd.read_csv("2-Intro to data structures/baseball.csv")
print(baseball)
baseball.info()

# However, using DataFrame.to_string() will return a string representation of the DataFrame
# in tabular form, though it won’t always fit the console width:

print(baseball.iloc[-20:, :12].to_string())

# Wide DataFrames will be printed across multiple rows by default:

print(pd.DataFrame(np.random.randn(3, 12)))

# You can change how much to print on a single row by setting the display.width option:

pd.set_option("display.width", 40)  # default is 80

print(pd.DataFrame(np.random.randn(3, 12)))

# You can adjust the max width of the individual columns by setting display.max_colwidth
datafile = {
    "filename": ["filename_01", "filename_02"],
    "path": [
        "media/user_name/storage/folder_01/filename_01",
        "media/user_name/storage/folder_02/filename_02",
    ],
}

pd.set_option("display.max_colwidth", 30)
print(pd.DataFrame(datafile))
pd.set_option("display.max_colwidth", 100)
print(pd.DataFrame(datafile))

# You can also disable this feature via the expand_frame_repr option. This will print the table
# in one block.

##############################################################
# DataFrame column attribute access and IPython completion
##############################################################

# If a DataFrame column label is a valid Python variable name, the column can be accessed like an attribute:
df = pd.DataFrame({"foo1": np.random.randn(5), "foo2": np.random.randn(5)})
print(df)
print(df.foo1)

# The columns are also connected to the IPython completion mechanism so they can be tab-completed:
# df.foo<TAB>  # noqa: E225, E999
# df.foo1  df.foo2
