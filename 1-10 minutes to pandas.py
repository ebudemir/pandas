# https://pandas.pydata.org/docs/user_guide/10min.html
##############################################################
# 10 minutes to pandas
##############################################################

# This is a short introduction to pandas, geared mainly for new users. You can see more complex recipes
# in the Cookbook. https://pandas.pydata.org/docs/user_guide/cookbook.html#cookbook
#
# Customarily, we import as follows:

import pandas as pd
import numpy as np

##############################################################
# Basic data structures in pandas
##############################################################
# Pandas provides two types of classes for handling data:
#
# Series: a one-dimensional labeled array holding data of any type
# such as integers, strings, Python objects etc.
#
# DataFrame: a two-dimensional data structure that holds data like a two-dimension array or a
# table with rows and columns.

##############################################################
# Object creation
##############################################################
#
# See the Intro to data structures section.
#
# Creating a Series by passing a list of values, letting pandas create a default RangeIndex.

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# Creating a DataFrame by passing a NumPy array with a datetime index using date_range()
# and labeled columns:

dates = pd.date_range("20130101", periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)

# Creating a DataFrame by passing a dictionary of objects where the keys are the column labels
# and the values are the column values.

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)

print(df2)

# The columns of the resulting DataFrame have different dtypes:
print(df2.dtypes)

# If you’re using IPython, tab completion for column names (as well as public attributes) is automatically
# enabled. Here’s a subset of the attributes that will be completed:
print(df2.B)

# As you can see, the columns A, B, C, and D are automatically tab completed. E and F are there as well;
# the rest of the attributes have been truncated for brevity.

##############################################################
# Viewing data
##############################################################
# See the Essentially basics functionality section.
# Use DataFrame.head() and DataFrame.tail() to view the top and bottom rows of the frame respectively:
print(df.head())
print(df.tail(3))

# Display the DataFrame.index or DataFrame.columns:
print(df.index)
print(df.columns)

# Return a NumPy representation of the underlying data with DataFrame.to_numpy() without the index
# or column labels:
print(df.to_numpy())
print(df.to_numpy().dtype)

# NumPy arrays have one dtype for the entire array while pandas DataFrames have one dtype per column.
# When you call DataFrame.to_numpy(), pandas will find the NumPy dtype that can hold all of the dtypes
# in the DataFrame. If the common data type is object, DataFrame.to_numpy() will require copying data.
print(df2.dtypes)
print(df2.to_numpy())
print(df2.to_numpy().dtype)

# describe() shows a quick statistic summary of your data:
print(df.describe())

# Transposing your data:
print(df.T)

# DataFrame.sort_index() sorts by an axis:
print(df.sort_index(axis=1, ascending=False))

# DataFrame.sort_values() sorts by values:
print(df.sort_values(by="B", ascending=False))

##############################################################
# Selection
##############################################################
# While standard Python / NumPy expressions for selecting and setting are intuitive and come in handy
# for interactive work, for production code, we recommend the optimized pandas data access methods,
# DataFrame.at(), DataFrame.iat(), DataFrame.loc() and DataFrame.iloc().

# See the indexing documentation Indexing and Selecting Data and MultiIndex / Advanced Indexing.

# Getitem ([])
# For a DataFrame, passing a single label selects a columns and yields a Series equivalent to df.A:
print(df["A"])

# For a DataFrame, passing a slice : selects matching rows:
print(df)

# including starting index excluding ending index
print(df[0:3])

# this case including starting and ending index
print(df["20130102":"20130104"])

# Selection by label
# See more in Selection by Label using DataFrame.loc() or DataFrame.at().
# Selecting a row matching a label:
print(df.loc[dates[0]])
print(dates[0])
print(type(df.loc[dates[0]]))

# Selecting all rows (:) with a select column labels:
print(df.loc[:, ["A", "B"]])

# For label slicing, both endpoints are included:
print(df.loc["20130102":"20130104", ["A", "B"]])

# Selecting a single row and column label returns a scalar:
print(df.loc[dates[0], "A"])

# For getting fast access to a scalar (equivalent to the prior method):
print(df.at[dates[0], "A"])

# Selection by position
# See more in Selection by Position using DataFrame.iloc() or DataFrame.iat().
# Select via the position of the passed integers:
print(df.iloc[3])
print(type(df.iloc[3]))

# Integer slices acts similar to NumPy/Python:
print(df.iloc[3:5, 0:2])
print(type(df.iloc[3:5, 0:2]))

# Lists of integer position locations:
print(df.iloc[[1, 2, 4], [0, 2]])
print(type(df.iloc[[1, 2, 4], [0, 2]]))

# For slicing rows explicitly:
print(df.iloc[1:3, :])
print(type(df.iloc[1:3, :]))

# For slicing columns explicitly:
print(df.iloc[:, 1:3])
print(type(df.iloc[:, 1:3]))

# For getting a value explicitly:
print(df.iloc[1, 1])
print(type(df.iloc[1, 1]))

# For getting fast access to a scalar (equivalent to the prior method):
print(df.iat[1, 1])

# Boolean indexing
# Select rows where df.A is greater than 0.
print(df)
print(df[df["A"] > 0])

# Selecting values from a DataFrame where a boolean condition is met:
print(df[df > 0])

# Using isin() method for filtering:
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
print(df2)
print(df2[df2["E"].isin(["two", "four"])])

# Setting
# Setting a new column automatically aligns the data by the indexes:
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
print(s1)
df["F"] = s1
print(df)

# Setting values by label:
df.at[dates[0], "A"] = 0
print(df)

# Setting values by position:
df.iat[0, 1] = 0
print(df)

# Setting by assigning with a NumPy array:
df.loc[:, "D"] = np.array([5] * len(df))
print(np.array([5] * len(df)))
# The result of the prior setting operations:
print(df)

# A where operation with setting:
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

##############################################################
# Missing data
##############################################################
# For NumPy data types, np.nan represents missing data. It is by default not included in computations.
# See the Missing Data section.
#
# Reindexing allows you to change/add/delete the index on a specified axis.
# This returns a copy of the data:
print(df)
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0]: dates[1], "E"] = 1
print(df1)

# DataFrame.dropna() drops any rows that have missing data:
print(df1.dropna(how="any"))

# DataFrame.fillna() fills missing data:
print(df1.fillna(value=5))

# isna() gets the boolean mask where values are nan:
print(df1)
print(pd.isna(df1))

# Operations
# See the Basic section on Binary Ops.

# Stats
# Operations in general exclude missing data.
#
# Calculate the mean value for each column:
print(df.mean(axis=0))

# Calculate the mean value for each row:
print(df.mean(axis=1))

# Operating with another Series or DataFrame with a different index or column will align the result
# with the union of the index or column labels. In addition, pandas automatically broadcasts
# along the specified dimension and will fill unaligned labels with np.nan.
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(dates)
print(df)
print(s)
print(df.sub(s, axis="index"))

# User defined functions
# DataFrame.agg() and DataFrame.transform() applies a user defined function
# that reduces or broadcasts its result respectively.
print(df)
print(df.agg(lambda x: np.mean(x) * 5.6))
print(type(df.agg(lambda x: np.mean(x) * 5.6)))
print(df.transform(lambda x: x + 1))

# Value Counts
# See more at Histogramming and Discretization.
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())

# String Methods
# Series is equipped with a set of string processing methods in the str attribute that make it easy to
# operate on each element of the array, as in the code snippet below. See more at Vectorized String Methods.
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
print(s.str.lower())

##############################################################
# Merge
##############################################################

# Concat
# pandas provides various facilities for easily combining together Series and DataFrame objects
# with various kinds of set logic for the indexes and relational algebra functionality
# in the case of join / merge-type operations.
#
# See the Merging section.
#
# Concatenating pandas objects together row-wise with concat():
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
pieces = [df[:3], df[3:7], df[7:]]
print(df[:3])
print(df[3:7])
print(df[7:])
print(pd.concat(pieces))

# Adding a column to a DataFrame is relatively fast. However, adding a row requires a copy, and may be
# expensive. We recommend passing a pre-built list of records to the DataFrame constructor
# instead of building a DataFrame by iteratively appending records to it.

# Join
# merge() enables SQL style join types along specific columns. See the Database style joining section.
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
print(left)
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
print(right)
print(pd.merge(left, right, on="key"))

# merge() on unique keys:
left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
print(left)
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})
print(right)
print(pd.merge(left, right, on="key"))

##############################################################
# Grouping
##############################################################
# By “group by” we are referring to a process involving one or more of the following steps:
# Splitting the data into groups based on some criteria
# Applying a function to each group independently
# Combining the results into a data structure
#
# See the Grouping section.

df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)
print(df)

# Grouping by a column label, selecting column labels, and then applying the DataFrameGroupBy.sum()
# function to the resulting groups:
print(df.groupby("A")[["C", "D"]].sum())

# Grouping by multiple columns label forms MultiIndex.
print(df.groupby(["A", "B"]).sum())

##############################################################
# Reshaping
##############################################################

# See the sections on Hierarchical Indexing and Reshaping.
# Stack

arrays = [
   ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
   ["one", "two", "one", "two", "one", "two", "one", "two"],
]
print(arrays)
index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
print(index)
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
print(df)
df2 = df[:4]
print(df2)

# The stack() method “compresses” a level in the DataFrame’s columns:
stacked = df2.stack(future_stack=True)
print(stacked)

# With a “stacked” DataFrame or Series (having a MultiIndex as the index), the inverse operation of stack()
# is unstack(), which by default unstacks the last level:
print(stacked.unstack())
print(stacked.unstack(1))
print(stacked.unstack(0))

# Pivot tables
# See the section on Pivot Tables.
df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)
print(df)

# pivot_table() pivots a DataFrame specifying the values, index and columns
print(pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"]))

##############################################################
# Time series
##############################################################

# pandas has simple, powerful, and efficient functionality for performing resampling operations
# during frequency conversion (e.g., converting secondly data into 5-minutely data).
# This is extremely common in, but not limited to, financial applications. See the Time Series section.

rng = pd.date_range("1/1/2012", periods=100, freq="s")
print(rng)

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts)

print(ts.resample("5Min").sum())

# Series.tz_localize() localizes a time series to a time zone:
rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
print(rng)
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)
ts_utc = ts.tz_localize("UTC")
print(ts_utc)

# Series.tz_convert() converts a timezones aware time series to another time zone:
print(ts_utc.tz_convert("US/Eastern"))

# Adding a non-fixed duration (BusinessDay) to a time series:
print(rng)
print(rng + pd.offsets.BusinessDay(5))

##############################################################
# Categoricals
##############################################################

# pandas can include categorical data in a DataFrame. For full docs, see the categorical introduction
# and the API documentation.

df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)
print(df)
df["grade"] = df["raw_grade"].astype("category")
print(df)
print(df.grade)
# Rename the categories to more meaningful names:
new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)
print(df)
print(df.grade)

# Reorder the categories and simultaneously add the missing categories
# (methods under Series.cat() return a new Series by default):
df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)
print(df.grade)

# Sorting is per order in the categories, not lexical order:
print(df.sort_values(by="grade"))

# Grouping by a categorical column with observed=False also shows empty categories:
print(df.groupby("grade", observed=False).size())

##############################################################
# Plotting
##############################################################
# See the Plotting docs.
#
# We use the standard convention for referencing the matplotlib API:
import matplotlib.pyplot as plt
plt.close("all")

# The plt.close method is used to close a figure window:
ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
print(ts)
ts = ts.cumsum()
print(ts)
#ts.plot()
#plt.show()

# When using Jupyter, the plot will appear using plot(). Otherwise use matplotlib.pyplot.show
# to show it or matplotlib.pyplot.savefig to write it to a file.

# plot() plots all columns:
df = pd.DataFrame(
    np.random.randn(1000, 4), index=ts.index, columns=["A", "B", "C", "D"]
)
df = df.cumsum()
#plt.figure()
df.plot()
plt.legend(loc='best')
#plt.show()

##############################################################
# Importing and exporting data
##############################################################

# See the IO Tools section.
# CSV
# Writing to a csv file: using DataFrame.to_csv()

df = pd.DataFrame(np.random.randint(0, 5, (10, 5)))
df.to_csv("1-10 minutes to pandas/foo.csv")

# Reading from a csv file: using read_csv()
print(pd.read_csv("1-10 minutes to pandas/foo.csv"))

# Parquet
# Writing to a Parquet file:
df.to_parquet("1-10 minutes to pandas/foo.parquet")

# Reading from a Parquet file Store using read_parquet():
print(pd.read_parquet("1-10 minutes to pandas/foo.parquet"))

# Excel
# Reading and writing to Excel.
# Writing to an excel file using DataFrame.to_excel():
df.to_excel("1-10 minutes to pandas/foo.xlsx", sheet_name="Sheet1")

# Reading from an excel file using read_excel():
print(pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"]))

# Gotchas
# If you are attempting to perform a boolean operation
# on a Series or DataFrame you might see an exception like:
# UNCOMMENT TO SEE THE EXCEPTION
# if pd.Series([False, True, False]):
#     print("I was true")

# See Comparisons and Gotchas for an explanation and what to do.


