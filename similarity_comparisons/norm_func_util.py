import numpy as np

from constants import *
from util_helpers import *


def np_norm_func_wrapper(df, np_norm_func, axis=1):
    """Wrapper for norm funcs that take a numpy array"""
    # get numpy array of data
    data_as_numpy = df.to_numpy(copy=True)
    if (axis == 0):
        # normalize columns instead of rows - transpose data
        data_as_numpy = np.transpose(data_as_numpy)
    normed_numpy = np_norm_func(data_as_numpy)
    if (axis == 0):
        # transpose back to original shape
        normed_numpy = np.transpose(normed_numpy)
    return pd.DataFrame(normed_numpy, columns=df.columns, index=df.index)


def truncate_epochs(df, cutoff=None):
    """Removes epochs greater than cutoff from DataFrame. Drop all columns with NaN values if no cutoff given. """
    # find the Index/axis that has epochs 
    pandas_index_to_use, axis, _, _ = find_level(EPOCH_INDEX_NAME, df.index, df.columns)
    # drop all NaN values if no cutoff given
    if (cutoff is None):
        df.dropna(axis=axis, inplace=True)
    else:
        # get all epochs
        all_values = pandas_index_to_use.unique(EPOCH_INDEX_NAME)
        # get the epochs that need to be removed from the DataFrame
        epochs_to_drop = [v for v in all_values if int(v) > cutoff]
        # drop unneeded DataFrames
        df.drop(epochs_to_drop, axis=axis, level=EPOCH_INDEX_NAME, inplace=True)
    return df


def norm_trajs(df, norm_traj_func):
    """Returns DataFrame with trajectories normalized with norm_traj_func"""
    # get dict of trajectories
    traj_dict, axis = get_groups_of_level(df, EPOCH_INDEX_NAME)
    normed_outer_dict = {}
    for name in traj_dict:
        if (isinstance(traj_dict[name], dict)):
            normed_inner_dict = {inner_name: norm_traj_func(traj_df, axis)
                                 for (inner_name, traj_df) in traj_dict[name].items()}
            normed_outer_dict[name] = pd.concat(normed_inner_dict, axis=axis)
        else:
            normed_outer_dict[name] = norm_traj_func(traj_dict[name], axis)
    normed_df = pd.concat(normed_outer_dict, axis=1 if (axis == 0) else 0)
    # restore row Index and column Index of original DataFrame
    normed_df.index = df.index
    normed_df.columns = df.columns
    return normed_df


def df_dist_func_wrapper(data_i, data_j, dist_func, index_obj=None):
    """Wrapper for dist funcs that take a DataFrame/Series --
    coverts data to Series"""
    df_i = pd.Series(data_i, index=index_obj)
    df_j = pd.Series(data_j, index=index_obj)
    return dist_func(df_i, df_j)


def euclid_dist(df_i, df_j):
    """calculates euclidean distance between two DataFrames"""
    return np.linalg.norm(df_i.to_numpy() - df_j.to_numpy())


def get_curve(df, curve_scale_func=None, use_index_as_x=True):
    """ get list of (x, y) tuples
    :param df: DataFrame containing curve values
    :param curve_scale_func: function which scales x values of curve, takes list of points in curve
    :param use_index_as_x: if True, try to use index of squeezed DataFrame as x values (default True)
    """
    df_as_series = df.squeeze()
    unscaled_curve = None
    get_x_via_enum = not use_index_as_x
    # use index values as x values of curve
    if (use_index_as_x):
        # drop all levels except innermost
        df_as_series = df_as_series.droplevel(list(range(0, df_as_series.index.nlevels - 1)))
        try:
            # get unscaled curve
            unscaled_curve = np.array([(int(epoch), perf) for (epoch, perf) in df_as_series.items()])
        except ValueError as ve:
            get_x_via_enum = True
    # enumerate values in df to get (x, y) pairs
    if (get_x_via_enum):
        unscaled_curve = np.array(list(enumerate(df.to_numpy())))
    # rescale curve if rescaling function given
    return unscaled_curve if (curve_scale_func is None) else curve_scale_func(unscaled_curve)


def curve_dist(df_i, df_j, curve_dist_func, curve_scale_func=None):
    """given two 1-column or 1-row DataFrames, convert each to a list
    ("curve") of points (x, y) using get_curve. Return the distance
    between the two curves calculated by curve_dist_func"""
    curve_i = get_curve(df_i, curve_scale_func=curve_scale_func)
    curve_j = get_curve(df_j, curve_scale_func=curve_scale_func)
    return curve_dist_func(curve_i, curve_j)


def get_series(df):
    """ get list of performance values in df """
    return df.squeeze().to_numpy()


def series_dist(df_i, df_j, series_dist_func):
    """ given two 1-column or 1-row DataFrames, get the performance values
    as a list ("series"). Return the distance between the two series
    calculated series_dist_func """
    series_i = get_series(df_i)
    series_j = get_series(df_j)
    return series_dist_func(series_i, series_j)


def traj_dist_summary(df_i, df_j, dist_func, curve_scale_func=None, summary_func=np.mean):
    # get dict of trajectories of df_i
    trajs_i, axis_i = get_groups_of_level(df_i, EPOCH_INDEX_NAME)
    # get dict of trajectories of df_j
    trajs_j, axis_j = get_groups_of_level(df_j, EPOCH_INDEX_NAME)
    # initialize list to hold distances between each trajectory from df_i and the corresponding trajectory from df_j
    dist_array = []
    # iterate through pairs of trajectories (one from df_i and the corresponding one from df_j)
    for name in trajs_i:
        try:
            # try making trajectories into curves and computing distance
            dist = curve_dist(trajs_i[name], trajs_j[name], dist_func, curve_scale_func)
        except (TypeError, ValueError) as e:
            # dist_func does not take curves, try as series instead
            dist = series_dist(trajs_i[name], trajs_j[name], dist_func)
        # append distance to list of distances
        dist_array.append(dist)
    # return summary of distances
    return summary_func(dist_array)


def linear_curve_scale(unscaled_curve, new_xmin=None, new_xmax=None):
    if (new_xmin is None):
        new_xmin = np.min([y for (x, y) in unscaled_curve])
    if (new_xmax is None):
        new_xmax = np.max([y for (x, y) in unscaled_curve])
    new_xdiff = new_xmax - new_xmin
    old_xdiff = np.max([x for (x, y) in unscaled_curve]) - np.min([x for (x, y) in unscaled_curve])
    multiplier = new_xdiff / old_xdiff
    return np.array([(multiplier * x, y) for (x, y) in unscaled_curve])


# As of 10/29/20, switched to using numpy functions instead of pandas functions to do standardization.
# This results in a slight difference in the resulting values, as numpy.std uses the biased estimator
# (divide by vector length).
# https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy#:~:text=Pandas%20uses%20the%20unbiased%20estimator,std()%20.
def stdized_single_traj(traj_df, axis):
    """standardize trajectory"""
    traj_numpy = traj_df.to_numpy()
    traj_mean = np.mean(traj_numpy, axis=axis)
    traj_std = np.std(traj_numpy, axis=axis)
    result_numpy = (traj_numpy - traj_mean) / traj_std
    result_df = pd.DataFrame(result_numpy)
    result_df.index = traj_df.index
    result_df.columns = traj_df.columns
    return result_df

def standardized_trajs(df):
    """truncate and standardize all trajectories in a DataFrame"""
    # only include up to epoch 50 for all trajectories
    # so all trajectories have the same number of epochs being considered
    df = truncate_epochs(df)
    result_df = norm_trajs(df, stdized_single_traj)
    return result_df

