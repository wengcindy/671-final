# helper functions for similarity_util.py
import os

import pandas as pd
import numpy as np

from constants import *
from similarity_constants import *

def get_source_fname_parts(source_fpath):
    # get filename without extension
    source_fname = os.path.split(source_fpath)[1].rsplit(".", 1)[0]
    fname_parts = source_fname.split(OUTER_SEP)
    # initialize file suffix
    source_desc = fname_parts[0]
    # get what each level in the data frame represents
    row_lvl_names = fname_parts[1].split(MID_SEP)[1:]
    col_lvl_names = fname_parts[2].split(MID_SEP)[1:]
    return source_desc, row_lvl_names, col_lvl_names

def parse_df_from_source(source_fpath):
    """ Parse DataFrame from raw data file, infer row and column level names from filename"""
    source_desc, row_lvl_names, col_lvl_names = get_source_fname_parts(source_fpath)
    # get range of indices in source file containing row/col names
    row_name_inds = list(range(len(row_lvl_names)))
    col_name_inds = list(range(len(col_lvl_names)))

    # read in data from source_fpaths
    df = pd.read_csv(source_fpath, index_col=row_name_inds, header=col_name_inds)
    # set axis names
    df.index.set_names(row_lvl_names, inplace=True)
    df.columns.set_names(col_lvl_names, inplace=True)
    return df

def parse_df_from_result(source_fpath, num_row_levels=1, num_col_levels=1):
    """ Parse DataFrame from intermediate result data file"""
    # get range of indices in source file containing row/col names
    row_name_inds = list(range(num_row_levels))
    col_name_inds = list(range(num_col_levels))
    # read in data from source_fpaths
    df = pd.read_csv(source_fpath, index_col=row_name_inds, header=col_name_inds)
    return df


def get_used_vals_lvl_str(level_name, inc_exc_dict):
    """get string of values used for level with name level_name"""
    # key is either INC_KEY or EXC_KEY
    result = list(inc_exc_dict.keys())[0] + INNER_SEP + level_name + MID_SEP
    # only a one value in the dict, may be a list or a single value
    values = list(inc_exc_dict.values())[0]
    # convert values to strings, concatenate values onto result
    # can't pass a single string to join, the characters get separated
    result += MID_SEP.join(map(str, values)) if isinstance(values, list) else str(values)
    return result


def get_used_vals_strs(level_names, use_vals_dict):
    """helper method to get string representing values
    used/dropped from pandas_Index, for use in filename"""
    if ((INC_KEY in use_vals_dict) or (EXC_KEY in use_vals_dict)):
        # no levels specified, just use name of outermost level and return string
        return get_used_vals_lvl_str(level_names[0], use_vals_dict)
    else:
        # initialize list of strings for each level
        level_strs = []
        for level_key in use_vals_dict:
            # get the string level name if an integer index is used as key
            level_name = level_key if (not isinstance(level_key, int)) else level_names[level_key]
            # get inc_exc_dict for level
            level_dict = use_vals_dict[level_key]
            # append string for this level to list
            level_strs.append(get_used_vals_lvl_str(level_name, level_dict))
        return level_strs


def get_names_from_callables(lst):
    return [elm.__name__ if callable(elm) else elm for elm in lst]

def get_fname_from_parts(other_parts_dict, source_fpaths, use_rows=None, use_cols=None, extension=".csv"):
    fname_parts = []
    for key in OTHER_PARTS_KEY_ORDER:
        part = other_parts_dict.get(key)
        if (part is None):
            if (key in OTHER_PARTS_DEF_DICT):
                part = OTHER_PARTS_DEF_DICT[key]
            else:
                continue
        if isinstance(part, list):
            part = MID_SEP.join(get_names_from_callables(part))
        elif callable(part):
            part = part.__name__
        fname_parts.append(part)
    if (not isinstance(source_fpaths, list)):
        source_fpaths = [source_fpaths]
    fname_parts += [get_source_fname_parts(fpath)[0] for fpath in source_fpaths]

    _, row_lvl_names, col_lvl_names = get_source_fname_parts(source_fpaths[0])
    if (use_rows is not None):
        fname_parts += get_used_vals_strs(row_lvl_names, use_rows)
    if (use_cols is not None):
        fname_parts += get_used_vals_strs(col_lvl_names, use_cols)

    if (extension is None):
        extension = ""
    if (len(extension) > 0 and extension[0] != "."):
        extension = "." + extension
    return OUTER_SEP.join(fname_parts) + extension


def drop_level_vals(df, inc_exc_dict, axis, level=None):
    """helper method to drop values from level of pandas Index of df (in place)
    Arguments:
    df -- pandas DataFrame from which to drop values
    inc_exc_dict -- dict with either "include" or "exclude" key
        and value which is a list of a subset of row names 
        or of column names
    axis -- 1 to drop from column Index, 0 to drop from row Index
    level -- level of pandas_Index from which to get list of all 
        names - reference via int or name (default None - outermost level)
    """
    to_drop = []
    if (INC_KEY in inc_exc_dict):
        # dict specifies values to include, drop everything else from level
        # figure out which Index object to get all level values from
        pandas_Index = df.columns if (axis == 1) else df.index

        get_vals_from_level = 0 if (level is None) else level
        all_vals_in_level = pandas_Index.unique(get_vals_from_level)
        # get all vals not in inc_exc_dict["include"] to drop them
        to_drop = [n for n in all_vals_in_level
                   if (n not in inc_exc_dict[INC_KEY])]
    elif (EXC_KEY in inc_exc_dict):
        # dict specifies values to exclude, drop these from level
        to_drop = inc_exc_dict[EXC_KEY]
    # drop values
    df.drop(to_drop, axis=axis, level=level, inplace=True)


def drop_from_axis(df, use_vals_dict, axis):
    """helper method to drop values from pandas Index of df (in place)
    Arguments:
    df -- pandas DataFrame from which to drop values
    use_vals_dict -- inc_exc_dict (see drop_level_vals) or dict where keys 
        are level indices or names, and values are inc_exc_dicts
    axis -- 1 to drop from column Index, 0 to drop from row Index
    """
    if ((INC_KEY in use_vals_dict) or (EXC_KEY in use_vals_dict)):
        # no levels specified, drop values from outermost level
        drop_level_vals(df, use_vals_dict, axis)
    else:
        # iterate through levels
        for level in use_vals_dict:
            # only specify level if dropping from MultiIndex, to avoid errors
            if (((axis == 0) and (type(df.index).__name__ == "MultiIndex")) or
                    ((axis == 1) and (type(df.columns).__name__ == "MultiIndex"))):
                drop_level_vals(df, use_vals_dict[level], axis, level=level)
            else:
                drop_level_vals(df, use_vals_dict[level], axis)


def find_level(level_name, row_index, col_index=None):
    """Return, in this order, whichever of pandas Index objects row_index and col_index
    has level level_name, corresponding axis integer, level number, and Index
    object which does not have level name"""
    if (level_name in row_index.names):
        return row_index, 0, row_index.names.index(level_name), col_index
    elif ((col_index is not None) and (level_name in col_index.names)):
        return col_index, 1, col_index.names.index(level_name), row_index
    # level_name not in DataFrame
    else:
        raise ValueError("DataFrame does not have " + level_name + " axis")

def get_groups_of_level(df, level_name):
    """Given DataFrame/Series df and level name level_name, get dict of groups in df,
     grouping by all levels above level level_name and, if df is a DataFrame, all levels on axis which
     does not have level level_name. For example, if grouping by epochs
     (level name EPOCH_INDEX_NAME) on a DataFrame which has row levels (obj, runSeed)
     and column levels (optim, epoch), group by obj, runSeed, and optim so each group
     is a trajectory (i.e. has values for multiple epochs for a particular objective,
     run, and optimizer. In this example, the resulting dict would have format
     {(<some objective>, <some run>): {<some optimizer>: <trajectory DataFrame>, ...}, ...}.
     Also returns axis integer; if 0, groups will be columns, if 1, groups will be rows"""
    # get the Index/axis that has level_name, the level number, and the other Index
    try:
        col_index_obj = df.columns
    except AttributeError as ae:
        col_index_obj = None
    pandas_index_to_use, axis, level_ind, other_index = find_level(level_name, df.index, col_index_obj)

    # get row/col groups (whichever axis does not have level_name)
    if (other_index is not None):
        grouped = df.groupby(other_index.names, axis=1 if (axis == 0) else 0)
    else:
        # workaround for when df is a Series
        grouped = [("dummy name", df)]
    # initialize top-level dict
    group_dict = {}
    # iterate through row/col groups
    for name, group in grouped:
        # row/col group contains multiple of the desired groups
        if (pandas_index_to_use.nlevels > 1):
            # split group into subgroups based on levels above level level_name
            inner_grouped = group.groupby(pandas_index_to_use.names[0:level_ind], axis=axis)
            # put subgroups into dict and make that dict a value in the top-level dict
            group_dict[name] = {inner_name: inner_group for (inner_name, inner_group) in inner_grouped}
        # row/col group is one of the desired groups, no need for subgroups
        else:
            # add group to top-level dict
            group_dict[name] = group
    # flatten dict if top level has only one key-value pair
    group_dict_outer_keys = list(group_dict.keys())
    if (len(group_dict_outer_keys) == 1):
        group_dict = group_dict[group_dict_outer_keys[0]]
    # return top-level dict and axis which had level level_name
    # (axis indicates whether groups will be a column or a row)
    return group_dict, axis


def find_first_val(df, val, ret_axis=1):
    for row_name in df.index:
        row = df.loc[row_name]
        for col_name in df.columns:
            if ((row[col_name] == val) or (np.isnan(val) and np.isnan(row[col_name]))):
                return col_name if ret_axis == 1 else row_name
    return None
