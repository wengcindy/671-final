# methods to standardize how data is stored

import os
import platform
from constants import *

def get_valid_long_path(rel_path):
    """For relative path that is long, if OS is Windows, modify path
    to avoid errors. Otherwise make no change
    """
    if (platform.system() == 'Windows'):
        long_path_prefix = r'\\?\ '.strip()
        csv_path = long_path_prefix + os.path.abspath(rel_path)
        return csv_path
    else:
        return rel_path

def write_source_data(df, result_dir, result_fname=None, fname_data=None):
    """Write DataFrame to csv, compute filename if not given
    Arguments:
        df -- DataFrame to write to csv file
        result_dir -- directory in which to save csv file
        result_fname -- desired filename - must include ".csv" (default None)
        fname_data -- info to include in filename, if result_fname not given.
            Can be string (just file description, row and column level names
            from DataFrame will also be added to filename) or list of format
            [<file description>, <list of row level names>, <list of column
            level names>]
    Returns:
        result_fname -- filename of resulting csv file

    """
    # assemble result filename if none is given
    # format (parts in brackets can be provided as elements of fname_data):
    # {Some descriptor of data}___rows__{what are the row levels}___cols__{what are the col levels}.csv
    if (result_fname is None):
        if (fname_data is None):
            raise ValueError("No filename or file descriptor provided")
        # fname_data is just file descriptor
        elif (isinstance(fname_data, str)):
            fname_data = [fname_data, df.index.names, df.columns.names]
        # else fname data is already a list of [<file descriptor>, <row level names>, <col level names>

        # join level names into string
        # file descriptor is already a string
        fname_data_strs = [fname_data[0], MID_SEP.join(fname_data[1]), MID_SEP.join(fname_data[2])]

        result_fname = SOURCE_NAME_FORMAT_STR.format(*fname_data_strs)
        result_fname = result_fname + ".csv"
    csv_path = os.path.join(result_dir, result_fname)
    # can't get Index names to work correctly, so not including them
    # https://stackoverflow.com/questions/25151443/when-saving-a-pandas-dataframe-to-csv-how-do-i-retain-the-columns-name
    df_to_write = df.copy(deep=True)
    df_to_write.index.set_names([None] * df_to_write.index.nlevels, inplace=True)
    df_to_write.columns.set_names([None] * df_to_write.columns.nlevels, inplace=True)
    df_to_write.to_csv(get_valid_long_path(csv_path))
    return result_fname