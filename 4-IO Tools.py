# https://pandas.pydata.org/docs/user_guide/io.html
##############################################################
# IO tools (text, CSV, HDF5, …)
##############################################################

# The pandas I/O API is a set of top level reader functions accessed like pandas.read_csv()
# that generally return a pandas object. The corresponding writer functions are object methods
# that are accessed like DataFrame.to_csv(). Below is a table containing available readers and writers.

# Format Type         Data Description            Reader              Writer
# text                CSV                         read_csv            to_csv
# text                Fixed-Width Text File       read_fwf            NA
# text                JSON                        read_json           to_json
# text                HTML                        read_html           to_html
# text                LaTeX                       Styler.to_latex     NA
# text                XML                         read_xml            to_xml
# text                Local clipboard             read_clipboard      to_clipboard
# binary              MS Excel                    read_excel          to_excel
# binary              OpenDocument                read_excel          NA
# binary              HDF5 Format                 read_hdf            to_hdf
# binary              Feather Format              read_feather        to_feather
# binary              Parquet Format              read_parquet        to_parquet
# binary              ORC Format                  read_orc            to_orc
# binary              Stata                       read_stata          to_stata
# binary              SAS                         read_sas            NA
# binary              SPSS                        read_spss           NA
# binary              Python Pickle Format        read_pickle         to_pickle
# SQL                 SQL                         read_sql            to_sql
# SQL                 Google BigQuery;:ref:read_gbq<io.bigquery>;:ref:to_gbq<io.bigquery>
# Here is an informal performance comparison for some of these IO methods.

# For examples that use the StringIO class, make sure you import it with from io import StringIO for Python 3.


##############################################################
# CSV & text files
##############################################################

# The workhorse function for reading text files (a.k.a. flat files) is read_csv().
# See the cookbook for some advanced strategies.

# Parsing options

# read_csv() accepts the following common arguments:

# Basic

# ARGUMENT: filepath_or_buffer : various
# Either a path to a file (a str, pathlib.Path, or py:py._path.local.LocalPath), URL (including http, ftp,
# and S3 locations), or any object with a read() method (such as an open file or StringIO).
#
# ARGUMENT: sep : str, defaults to ',' for read_csv(), \t for read_table()
# Delimiter to use. If sep is None, the C engine cannot automatically detect the separator, but the Python
# parsing engine can, meaning the latter will be used and automatically detect the separator by Python’s
# builtin sniffer tool, csv.Sniffer. In addition, separators longer than 1 character and different from
# '\s+' will be interpreted as regular expressions and will also force the use of the Python parsing engine.
# Note that regex delimiters are prone to ignoring quoted data. Regex example: '\\r\\t'.
#
# ARGUMENT: delimiter : str, default None
# Alternative argument name for sep.
#
# ARGUMENT: delim_whitespace : boolean, default False
# Specifies whether or not whitespace (e.g. ' ' or '\t') will be used as the delimiter. Equivalent to
# setting sep='\s+'. If this option is set to True, nothing should be passed in for the delimiter parameter.

# Column and index locations and names

# ARGUMENT: header : int or list of ints, default 'infer'
# Row number(s) to use as the column names, and the start of the data. Default behavior is to infer the column
# names: if no names are passed the behavior is identical to header=0 and column names are inferred from the
# first line of the file, if column names are passed explicitly then the behavior is identical to header=None.
# Explicitly pass header=0 to be able to replace existing names.
#
# The header can be a list of ints that specify row locations for a MultiIndex on the columns e.g. [0,1,3].
# Intervening rows that are not specified will be skipped (e.g. 2 in this example is skipped). Note that
# this parameter ignores commented lines and empty lines if skip_blank_lines=True, so header=0 denotes
# the first line of data rather than the first line of the file.
#
# ARGUMENT : names : array-like, default None
# List of column names to use. If file contains no header row, then you should explicitly pass header=None.
# Duplicates in this list are not allowed.
#
# index_colint, str, sequence of int / str, or False, optional, default None
# Column(s) to use as the row labels of the DataFrame, either given as string name or column index.
# If a sequence of int / str is given, a MultiIndex is used.
#
# index_col=False can be used to force pandas to not use the first column as the index, e.g. when you have
# a malformed file with delimiters at the end of each line.
#
# The default value of None instructs pandas to guess. If the number of fields in the column header row
# is equal to the number of fields in the body of the data file, then a default index is used. If it is larger,
# then the first columns are used as index so that the remaining number of fields in the body are equal to
# the number of fields in the header.
#
# The first row after the header is used to determine the number of columns, which will go into the index.
# If the subsequent rows contain less columns than the first row, they are filled with NaN.
#
# This can be avoided through usecols. This ensures that the columns are taken as is and the trailing
# data are ignored.
#
# ARGUMENT : usecols : list-like or callable, default None
# Return a subset of the columns. If list-like, all elements must either be positional (i.e. integer indices
# into the document columns) or strings that correspond to column names provided either by the user in names or
# inferred from the document header row(s). If names are given, the document header row(s) are not taken
# into account. For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
#
# Element order is ignored, so usecols=[0, 1] is the same as [1, 0]. To instantiate a DataFrame from data
# with element order preserved use pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']] for columns
# in ['foo', 'bar'] order or pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']] for ['bar', 'foo'] order.
#
# If callable, the callable function will be evaluated against the column names, returning names where
# the callable function evaluates to True:

import pandas as pd
from io import StringIO

data = "col1,col2,col3\na,b,1\na,b,2\nc,d,3"
print(pd.read_csv(StringIO(data)))
print(pd.read_csv(StringIO(data), usecols=lambda x: x.upper() in ["COL1", "COL3"]))

# Using this parameter results in much faster parsing time and lower memory usage when using the
# c engine. The Python engine loads the data first before deciding which columns to drop.

# General parsing configuration

# ARGUMENT : dtypeType : name or dict of column -> type, default None
# Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32, 'c': 'Int64'} Use str or
# object together with suitable na_values settings to preserve and not interpret dtype.
# If converters are specified, they will be applied INSTEAD of dtype conversion.
#
# Added in version 1.5.0: Support for defaultdict was added. Specify a defaultdict as input
# where the default determines the dtype of the columns which are not explicitly listed.

# ARGUMENT : dtype_backend : {“numpy_nullable”, “pyarrow”}, defaults to NumPy backed DataFrames
# Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays, nullable dtypes
# are used for all dtypes that have a nullable implementation when “numpy_nullable” is set,
# pyarrow is used for all dtypes if “pyarrow” is set.
#
# The dtype_backends are still experimential.
#
# Added in version 2.0.
#
# ARGUMENT : engine : {'c', 'python', 'pyarrow'}
# Parser engine to use. The C and pyarrow engines are faster, while the python engine is currently
# more feature-complete. Multithreading is currently only supported by the pyarrow engine.
#
# Added in version 1.4.0: The “pyarrow” engine was added as an experimental engine, and some
# features are unsupported, or may not work correctly, with this engine.
#
# ARGUMENT : converters : dict, default None
# Dict of functions for converting values in certain columns. Keys can either be integers or column labels.
#
# ARGUMENT : true_values : list, default None
# Values to consider as True.
#
# ARGUMENT : false_values : list, default None
# Values to consider as False.
#
# ARGUMENT : skipinitialspace : boolean, default False
# Skip spaces after delimiter.
#
# ARGUMENT : skiprows : list-like or integer, default None
# Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
#
# If callable, the callable function will be evaluated against the row indices, returning True
# if the row should be skipped and False otherwise:

data = "col1,col2,col3\na,b,1\na,b,2\nc,d,3"
print(pd.read_csv(StringIO(data)))
print(pd.read_csv(StringIO(data), skiprows=lambda x: x % 2 != 0))

# ARGUMENT : skipfooter : int, default 0
# Number of lines at bottom of file to skip (unsupported with engine=’c’).
#
# ARGUMENT : nrows : int, default None
# Number of rows of file to read. Useful for reading pieces of large files.
#
# ARGUMENT : low_memory : boolean, default True
# Internally process the file in chunks, resulting in lower memory use while parsing, but possibly
# mixed type inference. To ensure no mixed types either set False, or specify the type with
# the dtype parameter. Note that the entire file is read into a single DataFrame regardless,
# use the chunksize or iterator parameter to return the data in chunks. (Only valid with C parser)
#
# ARGUMENT : memory_map : boolean, default False
# If a filepath is provided for filepath_or_buffer, map the file object directly onto memory and
# access the data directly from there. Using this option can improve performance because there
# is no longer any I/O overhead.

# NA and missing data handling

# ARGUMENT : na_values : scalar, str, list-like, or dict, default None
# Additional strings to recognize as NA/NaN. If dict passed, specific per-column NA values.
# See na values const below for a list of the values interpreted as NaN by default.
#
# ARGUMENT : keep_default_na : boolean, default True
# Whether or not to include the default NaN values when parsing the data. Depending on
# whether na_values is passed in, the behavior is as follows:
# If keep_default_na is True, and na_values are specified, na_values is appended to the default NaN
# values used for parsing.
# If keep_default_na is True, and na_values are not specified, only the default NaN values are used
# for parsing.
# If keep_default_na is False, and na_values are specified, only the NaN values specified na_values
# are used for parsing.
# If keep_default_na is False, and na_values are not specified, no strings will be parsed as NaN.
# Note that if na_filter is passed in as False, the keep_default_na and na_values parameters
# will be ignored.
#
# ARGUMENT : na_filter : boolean, default True
# Detect missing value markers (empty strings and the value of na_values). In data without any NAs,
# passing na_filter=False can improve the performance of reading a large file.
#
# ARGUMENT : verbose : boolean, default False
# Indicate number of NA values placed in non-numeric columns.
#
# ARGUMENT : skip_blank_lines : boolean, default True
# If True, skip over blank lines rather than interpreting as NaN values.

# Datetime handling

# ARGUMENT : parse_dates : boolean or list of ints or names or list of lists or dict, default False.
# If True -> try parsing the index.
# If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column.
# If [[1, 3]] -> combine columns 1 and 3 and parse as a single date column.
# If {'foo': [1, 3]} -> parse columns 1, 3 as date and call result ‘foo’.

# A fast-path exists for iso8601-formatted dates.

# ARGUMENT : infer_datetime_format : boolean, default False
# If True and parse_dates is enabled for a column, attempt to infer the datetime format to
# speed up the processing.

# Deprecated since version 2.0.0: A strict version of this argument is now the default,
# passing it has no effect.

# ARGUMENT : keep_date_col : boolean, default False
# If True and parse_dates specifies combining multiple columns then keep the original columns.
#
# ARGUMENT : date_parser : function, default None
# Function to use for converting a sequence of string columns to an array of datetime instances.
# The default uses dateutil.parser.parser to do the conversion. pandas will try to call date_parser
# in three different ways, advancing to the next if an exception occurs: 1) Pass one or more arrays
# (as defined by parse_dates) as arguments; 2) concatenate (row-wise) the string values from
# the columns defined by parse_dates into a single array and pass that; and 3) call date_parser
# once for each row using one or more strings (corresponding to the columns defined by parse_dates)
# as arguments.

# Deprecated since version 2.0.0: Use date_format instead, or read in as object and then
# apply to_datetime() as-needed.




