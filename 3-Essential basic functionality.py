# https://pandas.pydata.org/docs/user_guide/basics.html
##############################################################
# Essential basic functionality
##############################################################

# Here we discuss a lot of the essential functionality common to the pandas data structures.
# To begin, let’s create some example objects like we did in the 10 minutes to pandas section:

import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

index = pd.date_range("1/1/2000", periods=8)
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=["A", "B", "C"])
print(index)
print(s)
print(df)

##############################################################
# Head and tail
##############################################################

# To view a small sample of a Series or DataFrame object, use the head() and tail() methods.
# The default number of elements to display is five, but you may pass a custom number.

long_series = pd.Series(np.random.randn(1000))
print(long_series.head())
print(long_series.tail())

##############################################################
# Attributes and underlying data
##############################################################

# pandas objects have a number of attributes enabling you to access the metadata
#
# shape: gives the axis dimensions of the object, consistent with ndarray
#
# Axis labels
# Series: index (only axis)
#
# DataFrame: index (rows) and columns
#
# Note, these attributes can be safely assigned to!

print(df[:2])
df.columns = [x.lower() for x in df.columns]
print(df)

# pandas objects (Index, Series, DataFrame) can be thought of as containers for arrays,
# which hold the actual data and do the actual computation. For many types,
# the underlying array is a numpy.ndarray. However, pandas and 3rd party libraries
# may extend NumPy’s type system to add support for custom arrays (see dtypes).
#
# To get the actual data inside a Index or Series, use the .array property

print(s.array)
print(s.index.array)

# array will always be an ExtensionArray. The exact details of what an ExtensionArray is and
# why pandas uses them are a bit beyond the scope of this introduction. See dtypes for more.
#
# If you know you need a NumPy array, use to_numpy() or numpy.asarray().

print(s.to_numpy())
print(np.asarray(s))
#
# When the Series or Index is backed by an ExtensionArray, to_numpy() may involve copying data
# and coercing values. See dtypes for more.
#
# to_numpy() gives some control over the dtype of the resulting numpy.ndarray.
# For example, consider datetimes with timezones. NumPy doesn’t have a dtype to represent
# timezone-aware datetimes, so there are two possibly useful representations:
#
# An object-dtype numpy.ndarray with Timestamp objects, each with the correct tz
#
# A datetime64[ns] -dtype numpy.ndarray, where the values have been converted to
# UTC and the timezone discarded
#
# Timezones may be preserved with dtype=object

ser = pd.Series(pd.date_range("2000", periods=2, tz="CET"))
print(repr(ser.to_numpy(dtype=object)))

# Or thrown away with dtype='datetime64[ns]'
print(repr(ser.to_numpy(dtype="datetime64[ns]")))

# Getting the “raw data” inside a DataFrame is possibly a bit more complex. When your DataFrame
# only has a single data type for all the columns, DataFrame.to_numpy() will return the underlying data:

print(df.to_numpy())

# If a DataFrame contains homogeneously-typed data, the ndarray can actually be modified in-place,
# and the changes will be reflected in the data structure. For heterogeneous data (e.g. some of
# the DataFrame’s columns are not all the same dtype), this will not be the case.
# The values attribute itself, unlike the axis labels, cannot be assigned to.

# When working with heterogeneous data, the dtype of the resulting ndarray will be chosen to
# accommodate all of the data involved. For example, if strings are involved, the result will be
# of object dtype. If there are only floats and integers, the resulting array will be of float dtype.

# In the past, pandas recommended Series.values or DataFrame.values for extracting the data from
# a Series or DataFrame. You’ll still find references to these in old code bases and online.
# Going forward, we recommend avoiding .values and using .array or .to_numpy().
# .values has the following drawbacks:
#
# When your Series contains an extension type, it’s unclear whether Series.values returns
# a NumPy array or the extension array. Series.array will always return an ExtensionArray,
# and will never copy data. Series.to_numpy() will always return a NumPy array, potentially
# at the cost of copying / coercing values.
#
# When your DataFrame contains a mixture of data types, DataFrame.values may involve copying
# data and coercing values to a common dtype, a relatively expensive operation.
# DataFrame.to_numpy(), being a method, makes it clearer that the returned NumPy array may
# not be a view on the same data in the DataFrame.

##############################################################
# Accelerated operations
##############################################################

# pandas has support for accelerating certain types of binary numerical and boolean operations
# using the numexpr library and the bottleneck libraries.
#
# These libraries are especially useful when dealing with large data sets, and provide large speedups.
# numexpr uses smart chunking, caching, and multiple cores. bottleneck is a set of specialized
# cython routines that are especially fast when dealing with arrays that have nans.
#
# Here is a sample (using 100 column x 100,000 row DataFrames):

# Operation           0.11.0 (ms)     Prior Version (ms)      Ratio to Prior
# df1 > df2           13.32           125.35                  0.1063
# df1 * df2           21.71           36.63                   0.5928
# df1 + df2           22.04           36.50                   0.6039

# You are highly encouraged to install both libraries. See the section Recommended Dependencies
# for more installation info.
#
# These are both enabled to be used by default, you can control this by setting the options:

pd.set_option("compute.use_bottleneck", False)
pd.set_option("compute.use_numexpr", False)

##############################################################
# Flexible binary operations
##############################################################

# With binary operations between pandas data structures, there are two key points of interest:
#
# Broadcasting behavior between higher- (e.g. DataFrame) and lower-dimensional (e.g. Series) objects.
#
# Missing data in computations.
#
# We will demonstrate how to manage these issues independently, though they can be handled simultaneously.

# Matching / broadcasting behavior

# DataFrame has the methods add(), sub(), mul(), div() and related functions radd(), rsub(),
# … for carrying out binary operations. For broadcasting behavior, Series input is of primary interest.
# Using these functions, you can use to either match on the index or columns via the axis keyword:

df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)
df = pd.DataFrame(
    {
        "one": pd.Series([1, 2, 3], index=["a", "b", "c"]),
        "two": pd.Series([4, 5, 6, 7], index=["a", "b", "c", "d"]),
        "three": pd.Series([8, 9, 10], index=["b", "c", "d"]),
    }
)
print(df)

row = df.iloc[1]
column = df["two"]
print(row)
print(column)
print(df.sub(row, axis="columns"))
print(df.sub(row, axis=1))
print(df.sub(column, axis="index"))
print(df.sub(column, axis=0))

# Furthermore you can align a level of a MultiIndexed DataFrame with a Series.
dfmi = df.copy()
dfmi.index = pd.MultiIndex.from_tuples(
    [(1, "a"), (1, "b"), (1, "c"), (2, "a")], names=["first", "second"]
)
print(dfmi)
print(dfmi.sub(column, axis=0, level="second"))

# Series and Index also support the divmod() builtin. This function takes the floor division and
# modulo operation at the same time returning a two-tuple of the same type as the left hand side.
# For example:

s = pd.Series(np.arange(10))
print(s)
div, rem = divmod(s, 3)
print(div)
print(rem)

idx = pd.Index(np.arange(10))
print(idx)
div, rem = divmod(idx, 3)
print(div)
print(rem)

# We can also do elementwise divmod():
div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
print(div)
print(rem)

div, rem = divmod(idx, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
print(div)
print(rem)

# Missing data / operations with fill values
# In Series and DataFrame, the arithmetic functions have the option of inputting a fill_value,
# namely a value to substitute when at most one of the values at a location are missing.
# For example, when adding two DataFrame objects, you may wish to treat NaN as 0 unless
# both DataFrames are missing that value, in which case the result will be NaN
# (you can later replace NaN with some other value using fillna if you wish).

df2 = df.copy()
df2.loc["a", "three"] = 1.0
print(df)
print(df2)
print(df + df2)
print(df.add(df2, fill_value=0))

# Flexible comparisons

# Series and DataFrame have the binary comparison methods eq, ne, lt, gt, le, and ge
# whose behavior is analogous to the binary arithmetic operations described above:

print(df.gt(df2))
print(df2.ne(df))

# These operations produce a pandas object of the same type as the left-hand-side input that is
# of dtype bool. These boolean objects can be used in indexing operations, see the section
# on Boolean indexing.

# Boolean reductions

# You can apply the reductions: empty, any(), all(), and bool() to provide a way to summarize
# a boolean result.

print((df > 0).all())
print((df > 0).any())

# You can reduce to a final boolean value.

print((df > 0).any().any())

# You can test if a pandas object is empty, via the empty property.

print(df.empty)
print(pd.DataFrame(columns=list("ABC")).empty)

# Asserting the truthiness of a pandas object will raise an error, as the testing of
# the emptiness or values is ambiguous.

# UNCOMMENT TO SEE THE EXCEPTION
# if df:
#    print(True)
# df and df2

# Comparing if objects are equivalent

# Often you may find that there is more than one way to compute the same result. As a simple example,
# consider df + df and df * 2. To test that these two computations produce the same result,
# given the tools shown above, you might imagine using (df + df == df * 2).all().
# But in fact, this expression is False:

print(df + df == df * 2)
print((df + df == df * 2).all())

# Notice that the boolean DataFrame df + df == df * 2 contains some False values!
# This is because NaNs do not compare as equals:

print(np.nan == np.nan)

# So, NDFrames (such as Series and DataFrames) have an equals() method for testing equality,
# with NaNs in corresponding locations treated as equal.

print((df + df).equals(df * 2))

# Note that the Series or DataFrame index needs to be in the same order for equality to be True:
df1 = pd.DataFrame({"col": ["foo", 0, np.nan]})
df2 = pd.DataFrame({"col": [np.nan, 0, "foo"]}, index=[2, 1, 0])
print(df1)
print(df2)
print(df1.equals(df2))
print(df1.equals(df2.sort_index()))

# Comparing array-like objects

# You can conveniently perform element-wise comparisons when comparing
# a pandas data structure with a scalar value:

print(pd.Series(["foo", "bar", "baz"]) == "foo")
print(pd.Index(["foo", "bar", "baz"]) == "foo")

# pandas also handles element-wise comparisons between different array-like objects of the same length:
print(pd.Series(["foo", "bar", "baz"]) == pd.Index(["foo", "bar", "qux"]))
print(pd.Series(["foo", "bar", "baz"]) == np.array(["foo", "bar", "qux"]))

# Trying to compare Index or Series objects of different lengths will raise a ValueError:
# UNCOMMENT TO SEE THE EXCEPTION
# pd.Series(['foo', 'bar', 'baz']) == pd.Series(['foo', 'bar'])

# Combining overlapping data sets

# A problem occasionally arising is the combination of two similar data sets where values in
# one are preferred over the other. An example would be two data series representing a particular
# economic indicator where one is considered to be of “higher quality”. However, the lower
# quality series might extend further back in history or have more complete data coverage.
# As such, we would like to combine two DataFrame objects where missing values in one
# DataFrame are conditionally filled with like-labeled values from the other DataFrame.
# The function implementing this operation is combine_first(), which we illustrate:

df1 = pd.DataFrame(
    {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
)
df2 = pd.DataFrame(
    {"A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0], "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0]}
)
print(df1)
print(df2)
print(df1.combine_first(df2))


# General DataFrame combine

# The combine_first() method above calls the more general DataFrame.combine(). This method takes
# another DataFrame and a combiner function, aligns the input DataFrame and then passes the
# combiner function pairs of Series (i.e., columns whose names are the same).
#
# So, for instance, to reproduce combine_first() as above:

def combiner(x, y):
    return np.where(pd.isna(x), y, x)


print(df1.combine(df2, combiner))

##############################################################
# Descriptive statistics
##############################################################

# There exists a large number of methods for computing descriptive statistics and other related
# operations on Series, DataFrame. Most of these are aggregations (hence producing a lower-dimensional
# result) like sum(), mean(), and quantile(), but some of them, like cumsum() and cumprod(),
# produce an object of the same size. Generally speaking, these methods take an axis argument,
# just like ndarray.{sum, std, …}, but the axis can be specified by name or integer:
#
# Series: no axis argument needed
#
# DataFrame: “index” (axis=0, default), “columns” (axis=1)
#
# For example:

print(df)
print(df.mean(0))
print(df.mean(1))

# All such methods have a skipna option signaling whether to exclude missing data (True by default):

print(df.sum(0, skipna=False))
print(df.sum(axis=0, skipna=True))
print(df.sum(axis=1, skipna=False))
print(df.sum(axis=1, skipna=True))

# Combined with the broadcasting / arithmetic behavior, one can describe various statistical procedures,
# like standardization (rendering data zero mean and standard deviation of 1), very concisely:

print(df)
print(df.mean())
print(df.std())
ts_stand = (df - df.mean()) / df.std()
print(ts_stand)
print(ts_stand.std())
xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)
print(xs_stand)
print(xs_stand.std(1))

# Note that methods like cumsum() and cumprod() preserve the location of NaN values. This is somewhat
# different from expanding() and rolling() since NaN behavior is furthermore dictated by a min_periods
# parameter.
print(df)
print(df.cumsum())

# Here is a quick reference summary table of common functions. Each also takes an optional
# level parameter which applies only if the object has a hierarchical index.

# Function        Description
# count           Number of non-NA observations
# sum             Sum of values
# mean            Mean of values
# median          Arithmetic median of values
# min             Minimum
# max             Maximum
# mode            Mode
# abs             Absolute Value
# prod            Product of values
# std             Bessel-corrected sample standard deviation
# var             Unbiased variance
# sem             Standard error of the mean
# skew            Sample skewness (3rd moment)
# kurt            Sample kurtosis (4th moment)
# quantile        Sample quantile (value at %)
# cumsum          Cumulative sum
# cumprod         Cumulative product
# cummax          Cumulative maximum
# cummin          Cumulative minimum

# Note that by chance some NumPy methods, like mean, std, and sum, will exclude NAs on Series
# input by default

print(np.mean(df["one"]))
print(np.mean(df["one"].to_numpy()))

# Series.nunique() will return the number of unique non-NA values in a Series:

series = pd.Series(np.random.randn(500))
series[20:500] = np.nan
series[10:20] = 5
print(series.nunique())

# Summarizing data: describe

# There is a convenient describe() function which computes a variety of summary statistics about a
# Series or the columns of a DataFrame (excluding NAs of course):

series = pd.Series(np.random.randn(1000))
series[::2] = np.nan
print(series.describe())

frame = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
frame.iloc[::2] = np.nan
print(frame.describe())

# You can select specific percentiles to include in the output:

print(series.describe(percentiles=[0.05, 0.25, 0.75, 0.95]))

# By default, the median is always included.

# For a non-numerical Series object, describe() will give a simple summary of the number of
# unique values and most frequently occurring values:

s = pd.Series(["a", "a", "b", "b", "a", "a", np.nan, "c", "d", "a"])
print(s.describe())

# Note that on a mixed-type DataFrame object, describe() will restrict the summary to include only
# numerical columns or, if none are, only categorical columns:

frame = pd.DataFrame({"a": ["Yes", "Yes", "No", "No"], "b": range(4)})
print(frame.describe())

# This behavior can be controlled by providing a list of types as include/exclude arguments.
# The special value all can also be used:

print(frame.describe(include=["object"]))
print(frame.describe(include=["number"]))
print(frame.describe(include="all"))

# That feature relies on select_dtypes. Refer to there for details about accepted inputs.

# Index of min/max values

# The idxmin() and idxmax() functions on Series and DataFrame compute the index labels with the
# minimum and maximum corresponding values:

s1 = pd.Series(np.random.randn(5))
print(s1)
print(s1.idxmin(), s1.idxmax())

df1 = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
print(df1)
print(df1.idxmin(axis=0))
print(df1.idxmax(axis=1))

# When there are multiple rows (or columns) matching the minimum or maximum value, idxmin()
# and idxmax() return the first matching index:

df3 = pd.DataFrame([2, 1, 1, 3, np.nan], columns=["A"], index=list("edcba"))
print(df3["A"].idxmin())

# idxmin and idxmax are called argmin and argmax in NumPy.

# Value counts (histogramming) / mode

# The value_counts() Series method computes a histogram of a 1D array of values. It can also be used
# as a function on regular arrays:

data = np.random.randint(0, 7, size=50)
print(data)
s = pd.Series(data)
print(s.value_counts())

# The value_counts() method can be used to count combinations across multiple columns.
# By default all columns are used but a subset can be selected using the subset argument.

data = {"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"]}
frame = pd.DataFrame(data)
print(frame.value_counts())
print(frame.value_counts(subset=["a"]))
print(frame.value_counts(subset=["b"]))

# Similarly, you can get the most frequently occurring value(s), i.e. the mode, of the values
# in a Series or DataFrame:

s5 = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])
print(s5.mode())

df5 = pd.DataFrame(
    {
        "A": np.random.randint(0, 7, size=50),
        "B": np.random.randint(-10, 15, size=50),
    }
)
print(df5.mode())

# Discretization and quantiling

# Continuous values can be discretized using the cut() (bins based on values) and qcut()
# (bins based on sample quantiles) functions:

arr = np.random.randn(20)
print(arr)
factor = pd.cut(arr, 4)
print(factor)
factor = pd.cut(arr, [-5, -1, 0, 1, 5])
print(factor)

# qcut() computes sample quantiles. For example, we could slice up some normally distributed
# data into equal-size quartiles like so:

arr = np.random.randn(30)
factor = pd.qcut(arr, [0, 0.25, 0.5, 0.75, 1])
print(factor)

# We can also pass infinite values to define the bins:

arr = np.random.randn(20)
factor = pd.cut(arr, [-np.inf, 0, np.inf])
print(factor)


##############################################################
# Function application
##############################################################

# To apply your own or another library’s functions to pandas objects, you should be aware of
# the three methods below. The appropriate method to use depends on whether your function expects
# to operate on an entire DataFrame or Series, row- or column-wise, or elementwise.
#
# Tablewise Function Application: pipe()
# Row or Column-wise Function Application: apply()
# Aggregation API: agg() and transform()
# Applying Elementwise Functions: map()

# Tablewise function application

# DataFrames and Series can be passed into functions. However, if the function needs to be called
# in a chain, consider using the pipe() method.

# First some setup:
def extract_city_name(df):
    """
    Chicago, IL -> Chicago for city_name column
    """
    df["city_name"] = df["city_and_code"].str.split(",").str.get(0)
    return df


def add_country_name(df, country_name=None):
    """
    Chicago -> Chicago-US for city_name column
    """
    col = "city_name"
    df["city_and_country"] = df[col] + country_name
    return df


df_p = pd.DataFrame({"city_and_code": ["Chicago, IL"]})
print(df_p)

# extract_city_name and add_country_name are functions taking and returning DataFrames.
#
# Now compare the following:

print(add_country_name(extract_city_name(df_p), country_name="US"))

# Is equivalent to:

print(df_p.pipe(extract_city_name).pipe(add_country_name, country_name="US"))

# pandas encourages the second style, which is known as method chaining. pipe makes it easy to use
# your own or another library’s functions in method chains, alongside pandas’ methods.
#
# In the example above, the functions extract_city_name and add_country_name each expected
# a DataFrame as the first positional argument. What if the function you wish to apply takes
# its data as, say, the second argument? In this case, provide pipe with a tuple of (callable,
# data_keyword). .pipe will route the DataFrame to the argument specified in the tuple.
#
# For example, we can fit a regression using statsmodels. Their API expects a formula first
# and a DataFrame as the second argument, data. We pass in the function, keyword pair (sm.ols, 'data')
# to pipe:

import statsmodels.formula.api as sm

bb = pd.read_csv("2-Intro to data structures/baseball.csv", index_col="id")
print(
    bb.query("h > 0")
    .assign(ln_h=lambda df: np.log(df.h))
    .pipe((sm.ols, "data"), "hr ~ ln_h + year + g + C(lg)")
    .fit()
    .summary()
)

# The pipe method is inspired by unix pipes and more recently dplyr and magrittr, which have
# introduced the popular (%>%) (read pipe) operator for R. The implementation of pipe here
# is quite clean and feels right at home in Python. We encourage you to view the source
# code of pipe().

# Row or column-wise function application

# Arbitrary functions can be applied along the axes of a DataFrame using the apply() method, which,
# like the descriptive statistics methods, takes an optional axis argument:

print(df)
print(df.apply(lambda x: np.mean(x)))
print(df.apply(lambda x: np.mean(x), axis=1))
print(df.apply(lambda x: x.max() - x.min()))
print(df.apply(np.cumsum))
print(df.apply(np.exp))

# The apply() method will also dispatch on a string method name.

print(df.apply("mean"))
print(df.apply("mean", axis=1))

# The return type of the function passed to apply() affects the type of the final output from
# DataFrame.apply for the default behaviour:
#
# If the applied function returns a Series, the final output is a DataFrame. The columns match
# the index of the Series returned by the applied function.
#
# If the applied function returns any other type, the final output is a Series.
#
# This default behaviour can be overridden using the result_type, which accepts three options:
# reduce, broadcast, and expand. These will determine how list-likes return values
# expand (or not) to a DataFrame.
#
# apply() combined with some cleverness can be used to answer many questions about a data set.
# For example, suppose we wanted to extract the date where the maximum value for each column occurred:

tsdf = pd.DataFrame(
    np.random.randn(1000, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=1000),
)

print(tsdf)
print(tsdf.apply(lambda x: x.idxmax()))


# You may also pass additional arguments and keyword arguments to the apply() method.


def subtract_and_divide(x, sub, divide=1):
    return (x - sub) / divide


df_udf = pd.DataFrame(np.ones((2, 2)))
print(df_udf)
print(df_udf.apply(subtract_and_divide, args=(5,), divide=3))

# Another useful feature is the ability to pass Series methods to carry out some Series operation
# on each column or row:

tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)
tsdf.iloc[3:7] = np.nan
print(tsdf)
print(tsdf.apply(pd.Series.interpolate))

# Finally, apply() takes an argument raw which is False by default, which converts each row or
# column into a Series before applying the function. When set to True, the passed function will
# instead receive an ndarray object, which has positive performance implications if you do not
# need the indexing functionality.

# Aggregation API

# The aggregation API allows one to express possibly multiple aggregation operations in a single
# concise way. This API is similar across pandas objects, see groupby API, the window API, and
# the resample API. The entry point for aggregation is DataFrame.aggregate(), or the alias
# DataFrame.agg().
#
# We will use a similar starting frame from above:

tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)
tsdf.iloc[3:7] = np.nan
print(tsdf)

# Using a single function is equivalent to apply(). You can also pass named methods as strings.
# These will return a Series of the aggregated output:

print(tsdf.agg(lambda x: np.sum(x)))
print(tsdf.agg("sum"))
print(tsdf.sum())

# Single aggregations on a Series this will return a scalar value:

print(tsdf["A"].agg("sum"))

# Aggregating with multiple functions

# You can pass multiple aggregation arguments as a list. The results of each of the passed functions
# will be a row in the resulting DataFrame. These are naturally named from the aggregation function.

print(tsdf.agg(["sum"]))

# Multiple functions yield multiple rows:

print(tsdf.agg(["sum", "mean"]))

# On a Series, multiple functions return a Series, indexed by the function names:

print(tsdf["A"].agg(["sum", "mean"]))

# Passing a lambda function will yield a <lambda> named row:

print(tsdf["A"].agg(["sum", lambda x: x.mean()]))


# Passing a named function will yield that name for the row:

def mymean(x):
    return x.mean()


print(tsdf["A"].agg(["sum", mymean]))

# Aggregating with a dict

# Passing a dictionary of column names to a scalar or a list of scalars, to DataFrame.agg allows
# you to customize which functions are applied to which columns. Note that the results are not
# in any particular order, you can use an OrderedDict instead to guarantee ordering.

print(tsdf.agg({"A": "mean", "B": "sum"}))

# Passing a list-like will generate a DataFrame output. You will get a matrix-like output of all
# of the aggregators. The output will consist of all unique functions. Those that are not noted
# for a particular column will be NaN:

print(tsdf.agg({"A": ["mean", "min"], "B": "sum"}))

# Custom describe

# With .agg() it is possible to easily create a custom describe function, similar to the
# built in describe function.

from functools import partial
q_25 = partial(pd.Series.quantile, q=0.25)
q_25.__name__ = "25%"
q_75 = partial(pd.Series.quantile, q=0.75)
q_75.__name__ = "75%"
print(tsdf.agg(["count", "mean", "std", "min", q_25, "median", q_75, "max"]))

# Transform API

# The transform() method returns an object that is indexed the same (same size) as the original.
# This API allows you to provide multiple operations at the same time rather than one-by-one.
# Its API is quite similar to the .agg API.
#
# We create a frame similar to the one used in the above sections.

tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)
tsdf.iloc[3:7] = np.nan
print(tsdf)

# Transform the entire frame. .transform() allows input functions as: a NumPy function, a string
# function name or a user defined function.

print(tsdf.transform(np.abs))
print(tsdf.transform("abs"))
print(tsdf.transform(lambda x: x.abs()))

# Here transform() received a single function; this is equivalent to a ufunc application.

print(np.abs(tsdf))

# Passing a single function to .transform() with a Series will yield a single Series in return.

print(tsdf["A"].transform(np.abs))

# Transform with multiple functions

# Passing multiple functions will yield a column MultiIndexed DataFrame. The first level will be
# the original frame column names; the second level will be the names of the transforming functions.

print(tsdf.transform([np.abs, lambda x: x + 1]))

# Transforming with a dict

# Passing a dict of functions will allow selective transforming per column.

print(tsdf)
print(tsdf.transform({"A": np.abs, "B": lambda x: x + 1}))

# Passing a dict of lists will generate a MultiIndexed DataFrame with these selective transforms.
print(tsdf.transform({"A": np.abs, "B": [lambda x: x + 1, "sqrt"]}))

# Applying elementwise functions
# Since not all functions can be vectorized (accept NumPy arrays and return another array or value),
# the methods map() on DataFrame and analogously map() on Series accept any Python function taking
# a single value and returning a single value. For example:

df4 = df.copy()
print(df4)


def f(x):
    return len(str(x))


print(df4["two"].map(f))
print(df4.map(f))

# Series.map() has an additional feature; it can be used to easily “link” or “map” values defined
# by a secondary series. This is closely related to merging/joining functionality:

s = pd.Series(
    ["six", "seven", "six", "seven", "five"], index=["a", "b", "c", "d", "e"]
)
t = pd.Series({"six": 6.0, "seven": 7.0})
print(s)
print(t)
print(s.map(t))

##############################################################
# Reindexing and altering labels
##############################################################

# reindex() is the fundamental data alignment method in pandas. It is used to implement nearly
# all other features relying on label-alignment functionality. To reindex means to conform
# the data to match a given set of labels along a particular axis. This accomplishes several things:
#
# Reorders the existing data to match a new set of labels
#
# Inserts missing value (NA) markers in label locations where no data for that label existed
#
# If specified, fill data for missing labels using logic (highly relevant to working with
# time series data)
#
# Here is a simple example:

s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print(s)
print(s.reindex(["e", "b", "f", "d"]))


# Here, the f label was not contained in the Series and hence appears as NaN in the result.
#
# With a DataFrame, you can simultaneously reindex the index and columns:

print(df)
print(df.reindex(index=["c", "f", "b"], columns=["three", "two", "one"]))

# Note that the Index objects containing the actual axis labels can be shared between objects.
# So if we have a Series and a DataFrame, the following can be done:

print(s)
rs = s.reindex(df.index)
print(rs)
print(rs.index is df.index)

# This means that the reindexed Series’s index is the same Python object as the DataFrame’s index.
#
# DataFrame.reindex() also supports an “axis-style” calling convention, where you specify
# a single labels argument and the axis it applies to.

print(df)
print(df.reindex(["c", "f", "b"], axis="index"))
print(df.reindex(["three", "two", "one", "nonecolumn"], axis="columns"))

# MultiIndex / Advanced Indexing is an even more concise way of doing reindexing.

# When writing performance-sensitive code, there is a good reason to spend some time becoming
# a reindexing ninja: many operations are faster on pre-aligned data. Adding two unaligned
# DataFrames internally triggers a reindexing step. For exploratory analysis you will hardly
# notice the difference (because reindex has been heavily optimized), but when CPU cycles
# matter sprinkling a few explicit reindex calls here and there can have an impact.

# Reindexing to align with another object

# You may wish to take an object and reindex its axes to be labeled the same as another object.
# While the syntax for this is straightforward albeit verbose, it is a common enough operation
# that the reindex_like() method is available to make this simpler:

print(df)
df2 = df.reindex(["a", "b", "c"], columns=["one", "two"])
df3 = df2 - df2.mean()
print(df2)
print(df2.mean())
print(df3)
print(df.reindex_like(df2))

# Aligning objects with each other with align

# The align() method is the fastest way to simultaneously align two objects. It supports
# a join argument (related to joining and merging):

# join='outer': take the union of the indexes (default)
# join='left': use the calling object’s index
# join='right': use the passed object’s index
# join='inner': intersect the indexes

# It returns a tuple with both of the reindexed Series:

s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
s1 = s[:4]
s2 = s[1:]
print("s",s)
print(s1)
print(s2)
print(s1.align(s2))
print(s1.align(s2, join="inner"))
print(s1.align(s2, join="left"))

# For DataFrames, the join method will be applied to both the index and the columns by default:
print(df.align(df2, join="inner"))

# You can also pass an axis option to only align on the specified axis:
print(df.align(df2, join="inner", axis=0))

# If you pass a Series to DataFrame.align(), you can choose to align both objects either
# on the DataFrame’s index or columns using the axis argument:
print(df)
print(df2.iloc[0])
print(df.align(df2.iloc[0], axis=1))

# Filling while reindexing

# reindex() takes an optional parameter method which is a filling method
# chosen from the following table:

# Method              Action
# pad / ffill         Fill values forward
# bfill / backfill    Fill values backward
# nearest             Fill from the nearest index value

# We illustrate these fill methods on a simple Series:

rng = pd.date_range("1/3/2000", periods=8)
ts = pd.Series(np.random.randn(8), index=rng)
ts2 = ts.iloc[[0, 3, 6]]
print(ts)
print(ts2)
print(ts2.reindex(ts.index))
print(ts2.reindex(ts.index, method="ffill"))
print(ts2.reindex(ts.index, method="bfill"))
print(ts2.reindex(ts.index, method="nearest"))

# These methods require that the indexes are ordered increasing or decreasing.
#
# Note that the same result could have been achieved using ffill (except for method='nearest')
# or interpolate:

print(ts2.reindex(ts.index).ffill())

# reindex() will raise a ValueError if the index is not monotonically increasing or decreasing.
# fillna() and interpolate() will not perform any checks on the order of the index.

# Limits on filling while reindexing

# The limit and tolerance arguments provide additional control over filling while reindexing.
# Limit specifies the maximum count of consecutive matches:

print(ts2.reindex(ts.index, method="ffill", limit=1))

# In contrast, tolerance specifies the maximum distance between the index and indexer values:

print(ts2.reindex(ts.index, method="ffill", tolerance="1 day"))

# Notice that when used on a DatetimeIndex, TimedeltaIndex or PeriodIndex, tolerance will
# coerced into a Timedelta if possible. This allows you to specify tolerance with appropriate strings.

# Dropping labels from an axis

# A method closely related to reindex is the drop() function. It removes a set of labels from an axis:
print(df)
print(df.drop(["a", "d"], axis=0))
print(df.drop(["one"], axis=1))

# Note that the following also works, but is a bit less obvious / clean:
print(df.reindex(df.index.difference(["a", "d"])))

# Renaming / mapping labels

# The rename() method allows you to relabel an axis based on some mapping (a dict or Series)
# or an arbitrary function.

print(s)
print(s.rename(str.upper))

# If you pass a function, it must return a value when called with any of the labels
# (and must produce a set of unique values). A dict or Series can also be used:

print(df)
print(df.rename(
    columns={"one": "foo", "two": "bar"},
    index={"a": "apple", "b": "banana", "d": "durian"},
))

# If the mapping doesn’t include a column/index label, it isn’t renamed. Note that extra labels
# in the mapping don’t throw an error.

# DataFrame.rename() also supports an “axis-style” calling convention, where you specify
# a single mapper and the axis to apply that mapping to.

print(df.rename({"one": "foo", "two": "bar"}, axis="columns"))
print(df.rename({"a": "apple", "b": "banana", "d": "durian"}, axis="index"))

# Finally, rename() also accepts a scalar or list-like for altering the Series.name attribute.

print(s.rename("scalar-name"))

# The methods DataFrame.rename_axis() and Series.rename_axis() allow specific names
# of a MultiIndex to be changed (as opposed to the labels).

df = pd.DataFrame(
    {"x": [1, 2, 3, 4, 5, 6], "y": [10, 20, 30, 40, 50, 60]},
    index=pd.MultiIndex.from_product(
        [["a", "b", "c"], [1, 2]], names=["let", "num"]
    ),
)
print(df)
print(df.rename_axis(index={"let": "abc"}))
print(df.rename_axis(index=str.upper))

##############################################################
# Iteration
##############################################################

# The behavior of basic iteration over pandas objects depends on the type. When iterating over a Series,
# it is regarded as array-like, and basic iteration produces the values. DataFrames follow the dict-like
# convention of iterating over the “keys” of the objects.
#
# In short, basic iteration (for i in object) produces:
#
# Series: values
#
# DataFrame: column labels
#
# Thus, for example, iterating over a DataFrame gives you the column names:

df = pd.DataFrame(
    {"col1": np.random.randn(3), "col2": np.random.randn(3)}, index=["a", "b", "c"]
)
print(df)
for col in df:
    print(col)

# pandas objects also have the dict-like items() method to iterate over the (key, value) pairs.
#
# To iterate over the rows of a DataFrame, you can use the following methods:

# iterrows(): Iterate over the rows of a DataFrame as (index, Series) pairs. This converts the rows to
# Series objects, which can change the dtypes and has some performance implications.
#
# itertuples(): Iterate over the rows of a DataFrame as namedtuples of the values. This is a lot faster
# than iterrows(), and is in most cases preferable to use to iterate over the values of a DataFrame.

# Iterating through pandas objects is generally slow. In many cases, iterating manually over the rows
# is not needed and can be avoided with one of the following approaches:
#
# Look for a vectorized solution: many operations can be performed using built-in methods or
# NumPy functions, (boolean) indexing, …
#
# When you have a function that cannot work on the full DataFrame/Series at once, it is better to
# use apply() instead of iterating over the values. See the docs on function application.
#
# If you need to do iterative manipulations on the values but performance is important, consider
# writing the inner loop with cython or numba. See the enhancing performance section for some
# examples of this approach.

# You should never modify something you are iterating over. This is not guaranteed to work in all cases.
# Depending on the data types, the iterator returns a copy and not a view, and writing to it will have
# no effect!
#
# For example, in the following case setting the value has no effect:

df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
for index, row in df.iterrows():
    row["a"] = 10
print(df)

# items

# Consistent with the dict-like interface, items() iterates through key-value pairs:
#
# Series: (index, scalar value) pairs
#
# DataFrame: (column, Series) pairs
#
# For example:

for label, ser in df.items():
    print(label)
    print(ser)

# iterrows

# iterrows() allows you to iterate through the rows of a DataFrame as Series objects. It returns
# an iterator yielding each index value along with a Series containing the data in each row:

for row_index, row in df.iterrows():
    print(row_index, row, sep="\n")

# Because iterrows() returns a Series for each row, it does not preserve dtypes across the rows
# (dtypes are preserved across columns for DataFrames). For example,

df_orig = pd.DataFrame([[1, 1.5]], columns=["int", "float"])
print(df_orig.dtypes)
row = next(df_orig.iterrows())[1]
print(row)

# All values in row, returned as a Series, are now upcasted to floats, also the original
# integer value in column x:

print(row["int"].dtype)
print(df_orig["int"].dtype)

# To preserve dtypes while iterating over the rows, it is better to use itertuples()
# which returns namedtuples of the values and which is generally much faster than iterrows().

