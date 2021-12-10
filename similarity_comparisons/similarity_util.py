import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.manifold.t_sne import TSNE

import dtaidistance.dtw_ndim_visualisation as dtwnvis
import dtaidistance.dtw_ndim as dtwn
import dtaidistance.dtw as dtw

from data_storage_util import *

from norm_func_util import *

# code to get data from crowded valley results without manually scraping

# must include \\?\ prefix to avoid errors on windows due to long file path
cwd_vly_results_path = r'\\?\C:\Users\jpmac\Documents\Descending-through-a-Crowded-Valley---Results\results'


def get_cwd_vly_data(budget, lr_sched, perf_summary_func, result_fname=None, obj_list=None, optim_list=None):
    """Get performance data from crowded valley results
    Arguments:
    budget -- Consider only results for this tuning budget
        - can be "large_budget", "mid_budget", or "oneshot"
    lr_sched -- Consider only results for this learning rate schedule 
        - can be "cosine", "cosine_wr", "ltr", or "none"
    perf_summary_func -- function which returns performance summary statistic 
        given path to results for a particular objective and optimizer
    result_fname -- optional, the name of the file in which the results should be stored
        - should end in .csv 
    obj_list -- optional - list of objectives for which to get results - if none
        is given, all objectives for which there are results are used
    optim_list -- optional - list of objectives for which to get results - if none
        is given, all objectives for which there are results are used
    """
    source_folder_path = os.path.join(cwd_vly_results_path, budget, lr_sched)
    # use all objectives if no list is given
    if (obj_list is None):
        obj_list = os.listdir(source_folder_path)
    # initialize list of DataFrames each holding data for a particular objective
    # across all optimizers
    obj_df_list = []
    # initialize variable to hold filename data from perf_summary_func
    # done outside loops so it can used later
    fname_data = None
    for objective in obj_list:
        # path to results for desired objective
        obj_res_path = os.path.join(source_folder_path, objective)
        # use all optimizers if no list is given
        if (optim_list is None):
            optim_list = os.listdir(obj_res_path)
        # initialize list of DataFrames each holding data for 
        # a particular optimizer and objective
        optim_df_list = []
        for optim in optim_list:
            # path to results for desired optimizer
            optim_res_path = os.path.join(obj_res_path, optim)
            # get summary stats for this optimizer, summary_df could be None
            summary_df, fname_data = perf_summary_func(optim_res_path)
            # add this optimizer DataFrame to list of optimizer DataFrames
            if (summary_df is not None):
                optim_df_list.append(summary_df)

        # check if there are optimizer DataFrames to concatenate
        if (len(optim_df_list) > 0):
            # concatenate optimizer DataFrames to create objective DataFrame
            # -- put optimizer DataFrames side-by-side and add an additional 
            # column level to distinguish data for different optimizers
            obj_df = pd.concat(optim_df_list, keys=optim_list, axis=1)
            level_for_set_names = 0 if obj_df.columns.nlevels > 1 else None
            obj_df.columns.set_names(OPTIM_INDEX_NAME, level=level_for_set_names, inplace=True)
            # if inner column level does not have multiple values
            # (i.e. no additional information is provided by the level)
            # remove the level
            ignore_inner_lvl = len(list(optim_df_list[0])) <= 1
            if (ignore_inner_lvl):
                obj_df.columns = obj_df.columns.get_level_values(0)
            # add this objective DataFrame to list of objective DataFrames
            obj_df_list.append(obj_df)

    # if there is no data, throw error!
    if (len(obj_df_list) == 0):
        raise ValueError("Summary function did not return DataFrame for any (objective, optimizer) pair")
    # concatenate objective DataFrames to create one DataFrame
    # -- stack objective DataFrames top-to-bottom and add an 
    # additional row level to distinguish data for different objectives
    df = pd.concat(obj_df_list, keys=obj_list)
    level_for_set_names = 0 if df.index.nlevels > 1 else None
    df.index.set_names(OBJ_INDEX_NAME, level=level_for_set_names, inplace=True)
    # if inner column level does not have multiple values
    # (i.e. no additional information is provided by the level)
    # remove the level
    ignore_inner_lvl = len(list(obj_df_list[0].index)) <= 1
    if (ignore_inner_lvl):
        df.index = df.index.get_level_values(0)
    # write data and return filename
    result_dir = os.path.join(".", "Crowded_Valley_results", budget, lr_sched)
    result_fname = write_source_data(df, result_dir, result_fname, fname_data)
    return df, result_fname


def best_perf_stat(optim_path, perf_metric, runs="col", epochs="row"):
    """return selection of data for all ten runs with the 
    best-performing hyperparameters of the given optimizer
    Arguments:
    optim_path -- path to results for the given optimizer
    perf_metric -- performance metric to scrape and summarize
        - can be "test_losses", "train_losses", "valid_losses",
        "test_accuracies", "train_accuracies", "valid_accuracies"
    runs -- can be None (average data across all runs),
        "row" (store individual data for each run as a 
        subset of a table row), or "col" (store individual data 
        for each run as a subset of a table column)
    epochs -- can be None (collect only data for the final epoch),
        "row" (collect the entire trajectory and store as a 
        subset of a table row), or "col" (collect the entire 
        trajectory and store as a subset of a table column)
    """
    # compute filename data
    # compute general descriptor for the data
    fprefix = "best_perf" + MID_SEP + perf_metric + MID_SEP
    fprefix += "avg_" if (runs is None) else INDIV_INDIC
    fprefix += "final" if (epochs is None) else "traj"
    # compute list of strings for row and for column describing what 
    # each level of the pandas Index for that axis represents
    row_lvl_names, col_lvl_names = [], []
    if (runs == "row"):
        col_lvl_names.append(RUN_INDEX_NAME)
    elif (runs == "col"):
        row_lvl_names.append(RUN_INDEX_NAME)
    if (epochs == "row"):
        col_lvl_names.append(EPOCH_INDEX_NAME)
    elif (epochs == "col"):
        row_lvl_names.append(EPOCH_INDEX_NAME)
    # just provide the file description - row and column level names will be in the DataFrame
    fname_data = fprefix

    run_df_dict = {}
    # search directories and files for this objective and optimizer
    for path, dirs, files in os.walk(optim_path):
        # found best performance - look at performance metric for all runs in this folder
        if (len(files) > 1):
            for file in files:
                # read in json file as pandas dataframe
                json_df = pd.read_json(os.path.join(path, file), orient="index")
                run_key = None
                # store data for this run
                run_df = None
                # iterate through key-value pairs in the json
                for row in json_df.itertuples():
                    # generate key to identify this run
                    if (row[0] == "random_seed"):
                        run_key = SEED_INDICATOR + str(row[1])
                    if (row[0] == perf_metric):
                        epoch_list = list(range(len(row[1])))
                        if (epochs is None):
                            # store performance for final epoch
                            run_df = pd.DataFrame([row[1][-1]])
                        elif (epochs == "row"):
                            # store performance for each epoch as a 1-row DataFrame
                            run_df = pd.DataFrame([{k: v for (k, v) in enumerate(row[1])}], columns=epoch_list)
                            run_df.columns.rename(EPOCH_INDEX_NAME, inplace=True)
                        elif (epochs == "col"):
                            # store performance for each epoch as a 1-column DataFrame
                            run_df = pd.DataFrame(row[1], index=epoch_list)
                            run_df.index.rename(EPOCH_INDEX_NAME, inplace=True)
                # add DataFrame for this run to dict, if there is data
                # not all objectives have all perf metrics 
                if (run_df is not None):
                    run_df_dict[run_key] = run_df
    # "un-zip" dict of DataFrames           
    run_keys, run_df_list = list(run_df_dict.keys()), list(run_df_dict.values())
    # if no data, return None instead of DataFrame
    if (len(run_df_list) == 0):
        return None, fname_data
    # compute axis along which to concatenate DataFrames
    # (and compute mean if runs is None)
    # ax = 1 will put run DataFrames side-by-side and compute a mean for each row
    # ax = 0 will put run DataFrames top-and-bottom and compute a mean for each column
    ax = 1 if (runs == "row" or (runs is None and epochs == "col")) else 0
    # concatenate run DataFrames using axis ax, constructing an additional level 
    # of run seeds in the appropriate pandas Index
    df = pd.concat(run_df_list, keys=run_keys, axis=ax)
    # give run level correct index name
    if (runs == "row"):
        df.columns.set_names(RUN_INDEX_NAME, level=0, inplace=True)
    elif (runs == "col"):
        df.index.set_names(RUN_INDEX_NAME, level=0, inplace=True)
    if (runs == "row" and run_df_list[0].shape[1] == 1):
        df = df.droplevel(df.columns.nlevels - 1, axis=1)
    elif (not runs == "row" and run_df_list[0].shape[0] == 1):
        df = df.droplevel(df.index.nlevels - 1)
    # compute mean across all runs
    if (runs is None):
        df = df.mean(axis=ax)

    return df, fname_data


def get_normed_df(source_fpaths, norm_func=None, use_rows=None, use_cols=None,
                  result_dir=None, fillna=False, **norm_func_kwargs):
    """ Get dataframe with normalized values and suggested file suffix. 
    Will write to file if result_dir is given.
    source_fpaths -- name of file containing performance data
    norm_func -- optional, used to normalize data for each objective 
    use_rows -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of row names 
    use_cols -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of column names 
    result_dir -- optional, directory in which to store plot result
        - filename constructed based on source filename, norm_func, 
        and included/excluded rows/columns
    **norm_func_kwargs -- keyword arguments for norm_func, like which 
        axis to normalize on
    """
    df_with_all = None
    if (isinstance(source_fpaths, list)):
        df_list = [parse_df_from_source(path) for path in source_fpaths]
        # will fix this later -- should be able to drop optimizers not common to all sources
        # common_outermost_columns = list(set([df.columns.unique() for df in df_and_fsuffix_list]))
        df_with_all = pd.concat(df_list)
    else:
        df_with_all = parse_df_from_source(source_fpaths)

    # create a copy from which to drop data
    df = df_with_all.copy(deep=True)
    # drop rows
    if (use_rows is not None): drop_from_axis(df, use_rows, 0)
    # drop columns
    if (use_cols is not None): drop_from_axis(df, use_cols, 1)

    # if no normalization, just copy df
    normed_df = df.copy(deep=True)

    # create new normalized dataframe if norm_func given
    if (norm_func is not None):
        normed_df = norm_func(df, **norm_func_kwargs)

    # fill in NaN values - propagate last numerical value forward along row
    if (fillna):
        normed_df.fillna(method='pad', inplace=True)

    # write data, if result_dir is given
    if (result_dir is not None):
        other_fname_parts = {NFUNC_NAME_KEY: norm_func}
        csv_fname = get_fname_from_parts(other_fname_parts, source_fpaths, use_cols=use_cols, use_rows=use_rows)
        csv_path = os.path.join(result_dir, csv_fname)
        normed_df.to_csv(get_valid_long_path(csv_path))
    return normed_df


def calc_dist(source_fpaths, norm_func=None, dist_func=euclid_dist, use_rows=None, use_cols=None, result_dir=None,
              group_by=None):
    """Get similarity calculations for each possible pair of optimizers
    across all objectives (euclidean distance between performance arrays)
    Return dataframe of results and suggested file suffix from get_normed_df
    Arguments:
    source_fpaths -- name of file (or list of names) containing performance data
    norm_func -- optional, used to normalize data for each objective 
        before similarity calculation
    dist_func -- optional, used to compute distance between two vectors 
        (default euclidean distance)
    use_rows -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of row names 
    use_cols -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of column names 
    result_dir -- optional, directory in which to store plot result
        - filename constructed based on source filename and 
        transformation info
    """
    normed_df = get_normed_df(source_fpaths, norm_func=norm_func, use_rows=use_rows, use_cols=use_cols)
    # get groups to compare with each other
    if (group_by is None):
        group_by = normed_df.index.names[-1]
    pandas_index_to_use, axis, level_ind, other_index = find_level(group_by, normed_df.index, normed_df.columns)
    grouped = normed_df.groupby(pandas_index_to_use.names[0:level_ind + 1])
    # initialize list of rows for similarity result
    dist_row_list = []
    for name_i, group_i in grouped:
        # initialize row for similarity result
        dist_row = []
        for name_j, group_j in grouped:
            # must iterate through all pairs of groups so dist_func can work correctly
            dist = dist_func(group_i, group_j)
            dist_row.append(dist)
        dist_row_list.append(dist_row)

    # create Index for DataFrame from group names
    new_index = [name for (name, _) in grouped]
    if (isinstance(new_index[0], tuple)):
        new_index = pd.MultiIndex.from_tuples(new_index)
    # create DataFrame for similarity result
    sim_df = pd.DataFrame(dist_row_list, columns=new_index, index=new_index)

    # write data, if result_dir is given
    if (result_dir is not None):
        other_fname_parts = {NFUNC_NAME_KEY: norm_func, RESULT_DESC_KEY: ["similarity", dist_func]}
        csv_fname = get_fname_from_parts(other_fname_parts, source_fpaths, use_rows=use_rows, use_cols=use_cols)
        csv_path = os.path.join(result_dir, csv_fname)
        sim_df.to_csv(get_valid_long_path(csv_path))
    return sim_df


def get_cluster_pct_table(source_fpath=None, df=None, result_fname=None, result_dir=None):
    """Convert cluster labelling results to percentage of points in cluster per objective"""
    if ((source_fpath is None) and (df is None)):
        raise ValueError('No source path or pandas dataframe specified')
    if ((result_dir is not None) and (result_fname is None)):
        raise ValueError('specified result directory but not file name')
    if (source_fpath is not None):
        # read in data from source_fpaths
        df = pd.read_csv(source_fpath, index_col=0)
    # dict of dicts where each inner dict is a row
    data_dict = {}
    # dict to hold total number of points for each objective
    total_pts_in_obj = {}
    # list of column names (cluster numbers)
    col_names = []
    # iterate through rows in df
    for row_name in df.index:
        # get objective for this point
        obj = row_name[0] if isinstance(row_name, tuple) else row_name
        # get which cluster this point is in
        cluster_num = df.loc[row_name, CLUSTER_LABEL_COL_HEADER]
        # first time seeing this cluster number, add to list of column names
        if (cluster_num not in col_names):
            col_names.append(cluster_num)

        if (obj not in data_dict):
            data_dict[obj] = {}
        if (cluster_num not in data_dict[obj]):
            data_dict[obj][cluster_num] = 0
        # increase number of points belonging to this cluster for this objective
        data_dict[obj][cluster_num] += 1

        if (obj not in total_pts_in_obj):
            total_pts_in_obj[obj] = 0
        # increase total number of points for this objective
        total_pts_in_obj[obj] += 1
    # for each objective, convert values in data_dict to percentages of total
    for obj in data_dict:
        for cluster_num in data_dict[obj]:
            pct = (float(data_dict[obj][cluster_num]) / total_pts_in_obj[obj]) * 100
            data_dict[obj][cluster_num] = pct

    row_names = list(data_dict.keys())
    data_list = list(data_dict.values())
    # order columns correctly
    col_names.sort()
    # create dataframe
    pct_df = pd.DataFrame(data_list, columns=col_names, index=row_names)
    # write data, if result_dir is given
    if (result_dir is not None):
        result_fpath = os.path.join(result_dir, result_fname)
        with open(get_valid_long_path(result_fpath), 'w') as result_file:
            pct_df.fillna('').to_markdown(buf=result_file, tablefmt="grid")
    return pct_df


def calc_clusters(source_fpaths, kmeans_inst, norm_func=None, use_rows=None, use_cols=None, get_pct=True,
                  result_dir=None):
    """Cluster performance data from Crowded Valley results.
    Return table of cluster labels for each data point
    and suggested file suffix, or table of percentage of 
    points in cluster per objective
    Arguments:
    source_fpaths -- path to file containing performance data
    kmeans_inst -- an instance of sklearn.cluster.KMeans
    norm_func -- optional, used to normalize data for each objective 
        before KMeans transformation
    use_rows -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of row names 
    use_cols -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of column names
    get_pct -- optional, produce table of percentage of points 
        in cluster per objective (default True)
    result_dir -- optional, directory in which to store plot result
        - filename constructed based on source filename and 
        transformation info
    """
    # normalize performance data found at source_fpaths
    normed_df = get_normed_df(source_fpaths, norm_func=norm_func, use_rows=use_rows, use_cols=use_cols,
                              fillna=True)
    # put all the points in a 2D array
    all_points_array = normed_df.to_numpy(copy=True)
    # do the KMeans clustering
    kmeans_inst.fit(all_points_array)
    # create DataFrame from labels array
    cluster_label_df = pd.DataFrame(kmeans_inst.labels_, columns=[CLUSTER_LABEL_COL_HEADER],
                                    index=normed_df.index)

    # write data, if result_dir is given
    if (result_dir is not None):
        # add number of clusters to file name
        k_str = "k{}".format(kmeans_inst.get_params()["n_clusters"])
        other_fname_parts = {NFUNC_NAME_KEY: norm_func}
        if (not get_pct):
            other_fname_parts[RESULT_DESC_KEY] = ["kmeans_labels", k_str]
            csv_fname = get_fname_from_parts(other_fname_parts, source_fpaths, use_rows=use_rows, use_cols=use_cols)
            csv_path = os.path.join(result_dir, csv_fname)
            cluster_label_df.to_csv(get_valid_long_path(csv_path))
            return cluster_label_df
        else:
            other_fname_parts[RESULT_DESC_KEY] = ["pct_in_cluster", k_str]
            result_fname = get_fname_from_parts(other_fname_parts, source_fpaths, use_rows=use_rows, use_cols=use_cols,
                                                extension=".md")
            return get_cluster_pct_table(df=cluster_label_df, result_fname=result_fname, result_dir=result_dir)


def tsne_vis_perf_array(source_fpaths, tsne_inst, norm_func=None, use_rows=None, use_cols=None,
                        df_dep_dist_func=None, dist_dir=None, result_dir=None):
    """Get visualization of performance data in 2D space using TSNE
    Arguments:
    source_fpaths -- path to file containing performance data
    tsne_inst -- an instance of sklearn.manifold.TSNE
    norm_func -- optional, used to normalize data for each objective 
        before visualization
    use_rows -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of row names 
    use_cols -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of column names 
    result_dir -- optional, directory in which to store plot result
        - filename constructed based on source filename and 
        transformation info
    """
    metric = tsne_inst.get_params()["metric"]
    if (dist_dir is None):
        # normalize performance data found at source_fpaths
        data_df = get_normed_df(source_fpaths, norm_func=norm_func, use_rows=use_rows, use_cols=use_cols)
        # Drop columns with NaN values so TSNE doesn't break
        data_df.dropna(axis=1, inplace=True)
        if (df_dep_dist_func is not None):
            # wrap the given distance function so it works on rows of the array passed to fit_transform()
            metric = lambda i, j: df_dist_func_wrapper(i, j, df_dep_dist_func, index_obj=data_df.columns)

    elif (df_dep_dist_func is not None):
        try:
            other_fname_parts = {NFUNC_NAME_KEY: norm_func, RESULT_DESC_KEY: ["similarity", df_dep_dist_func]}
            csv_fname = get_fname_from_parts(other_fname_parts, source_fpaths, use_rows=use_rows, use_cols=use_cols)
            csv_path = os.path.join(dist_dir, csv_fname)
            num_levels = 2 if (INDIV_INDIC in csv_fname) else 1
            data_df = parse_df_from_result(csv_path, num_row_levels=num_levels, num_col_levels=num_levels)
        except FileNotFoundError as e:
            data_df = calc_dist(source_fpaths, norm_func=norm_func, dist_func=df_dep_dist_func, use_rows=use_rows,
                                use_cols=use_cols, result_dir=dist_dir)

        metric = "precomputed"
    else:
        raise ValueError("dist_dir specified but no dist_func given")

    # make a copy of tsne_inst
    tsne_inst = TSNE(**tsne_inst.get_params())
    # add dist_func_wrap as the distance function to use
    tsne_inst.set_params(metric=metric)

    # put the data in a 2D array
    all_points_array = data_df.to_numpy(copy=True)
    # do the t-sne transformation
    embed_df = pd.DataFrame(tsne_inst.fit_transform(all_points_array), index=data_df.index, columns=["x", "y"])
    # get list of objectives
    objectives = embed_df.index.unique(0) if embed_df.index.nlevels > 1 else embed_df.index.unique()
    # set up figure
    fig = plt.figure(figsize=(12, 5))
    sub = fig.add_subplot(1, 1, 1)
    # display a scatter plot for each objective
    for obj in objectives:
        sub.scatter(embed_df.loc[obj]["x"], embed_df.loc[obj]["y"], label=obj)
    # display legend
    plt.legend()
    # save and show figure
    # list of relevant params as strings
    params_str_list = ["tsne_vis", "perp{}".format(tsne_inst.get_params()["perplexity"])]
    # get dist func name
    if (df_dep_dist_func is not None):
        dist_func_name = df_dep_dist_func.__name__
    else:
        dist_func_name = tsne_inst.get_params()["metric"] + "_dist"
    # decide whether to add dist func name to filename
    if (dist_func_name != "euclidean_dist"):
        params_str_list.append(dist_func_name)
    other_fname_parts = {NFUNC_NAME_KEY: norm_func, RESULT_DESC_KEY: params_str_list}
    img_fname = get_fname_from_parts(other_fname_parts, source_fpaths, use_rows=use_rows, use_cols=use_cols,
                                     extension=".png")
    fig.suptitle(img_fname)
    if (result_dir is not None):
        img_path = os.path.join(result_dir, img_fname)
        plt.savefig(get_valid_long_path(img_path))
    plt.show()
    return embed_df


def plot_perf_vs_optims(source_fpaths, norm_func=None, use_rows=None, use_cols=None, fgsize=(15, 15), result_dir=None):
    """Get plot of performance data where each line is an
        objective and each x-axis tick is a different optimizer

    Arguments:
    source_fpaths -- path to file containing performance data
    norm_func -- optional, used to normalize data for each objective 
        before plotting
    use_rows -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of row names 
    use_cols -- optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of column names 
    fgsize -- optional, (width, height) tuple for figsize
    result_dir -- optional, directory in which to store plot result
        - filename constructed based on source filename and 
        transformation info
    """
    # normalize performance data found at source_fpaths
    normed_df = get_normed_df(source_fpaths, norm_func=norm_func, use_rows=use_rows, use_cols=use_cols)
    # strip "Optimizer" to return shorter string for plotting        
    optims_short = [opt.replace("Optimizer", "") for opt in list(normed_df)]
    # set up figure, save, and show
    fig = plt.figure(figsize=fgsize)
    sub = fig.add_subplot(1, 1, 1)
    for obj in normed_df.index:
        # plot performance against optimizers for this objective
        sub.plot(optims_short, normed_df.loc[obj], label=obj)
    sub.legend()
    sub.set_facecolor('xkcd:white')
    fig.patch.set_facecolor('xkcd:white')
    plt.ylabel(get_fname_from_parts({NFUNC_NAME_KEY: norm_func}, source_fpaths, extension=None))
    # stagger ticks so labels don't overlap
    # https://stackoverflow.com/questions/51898101/how-do-i-stagger-or-offset-x-axis-labels-in-matplotlib
    for tick in sub.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    other_fname_parts = {RESULT_DESC_KEY: ["line_plot"], NFUNC_NAME_KEY: norm_func}
    img_fname = get_fname_from_parts(other_fname_parts, source_fpaths, use_rows=use_rows, use_cols=use_cols,
                                     extension=".png")
    fig.suptitle(img_fname)
    if (result_dir is not None):
        img_path = os.path.join(result_dir, img_fname)
        plt.savefig(img_path)
    plt.show()
    return normed_df


def plot_all_traj_dtw(source_fpaths, norm_func=None, use_rows=None, use_cols=None, curve_scale_func=None,
                      result_dir='.'):
    """THIS METHOD DOES NOT CURRENTLY PRODUCE INTELLIGIBLE RESULTS
    Plot dynamic time warping plots for all trajectories for all optimizers and objectives

    Arguments:
    :param source_fpaths: path to file containing performance data
    :param norm_func: optional, used to normalize data for each objective
        before plotting
    :param use_rows: optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of row names
    :param use_cols: optional, dict with either "include" or "exclude" key
        and value which is a list of a subset of column names
    :param curve_scale_func: function which scales the x-axis of all trajectories
    :param result_dir: directory in which to store plot results
        - full path constructed based on source filename and objectives
        being compared
    """
    # normalize performance data found at source_fpaths
    normed_df = get_normed_df(source_fpaths, norm_func=norm_func, use_rows=use_rows, use_cols=use_cols)
    for row_name_i in normed_df.index:
        rname_str_i = MID_SEP.join(row_name_i) if isinstance(row_name_i, tuple) else row_name_i
        for row_name_j in normed_df.index:
            rname_str_j = MID_SEP.join(row_name_j) if isinstance(row_name_j, tuple) else row_name_j
            mid_dir = get_fname_from_parts({NFUNC_NAME_KEY: norm_func}, source_fpaths, use_cols=use_cols,
                                           use_rows=use_rows, extension=None)
            file_dir = get_valid_long_path(os.path.join(result_dir, mid_dir, rname_str_i))
            os.makedirs(file_dir, exist_ok=True)
            file_path = os.path.join(file_dir, rname_str_j + ".pdf")
            with PdfPages(file_path) as pdf:
                for optim in normed_df.columns.unique(0):
                    traj_i = get_curve(normed_df.loc[row_name_i].loc[optim], curve_scale_func=curve_scale_func)
                    traj_j = get_curve(normed_df.loc[row_name_j].loc[optim], curve_scale_func=curve_scale_func)
                    dist, all_paths = dtwn.warping_paths(traj_i, traj_j)
                    best = dtw.best_path(all_paths)
                    fig, ax = dtwnvis.plot_warping(traj_i, traj_j, best)
                    plt.title(OUTER_SEP.join([rname_str_i, rname_str_j, optim, "dist_" + str(dist)]))
                    pdf.savefig(fig)
                    plt.close()
