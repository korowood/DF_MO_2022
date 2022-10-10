import pandas as pd


def create_decision(data: pd.DataFrame) -> list:
    # count
    cnt_pt = pd.pivot_table(data, values='decision_id', index=["user_id"],
                            columns=['period'], aggfunc="count").reset_index()

    mapper = {1: "cnt_1", 2: "cnt_2", 3: "cnt_3", 4: "cnt_4"}

    cnt_pt = cnt_pt.rename(columns=mapper)

    # max
    max_pt = pd.pivot_table(data, values='decision_id', index=["user_id"],
                            columns=['period'], aggfunc="max").reset_index()

    mapper = {1: "max_1", 2: "max_2", 3: "max_3", 4: "max_4"}

    max_pt = max_pt.rename(columns=mapper)

    # min
    min_pt = pd.pivot_table(data, values='decision_id', index=["user_id"],
                            columns=['period'], aggfunc="min").reset_index()

    mapper = {1: "min_1", 2: "min_2", 3: "min_3", 4: "min_4"}

    min_pt = min_pt.rename(columns=mapper)

    # mean
    mean_pt = pd.pivot_table(data, values='decision_id', index=["user_id"],
                             columns=['period'], aggfunc="mean").reset_index()

    mapper = {1: "mean_1", 2: "mean_2", 3: "mean_3", 4: "mean_4"}

    mean_pt = mean_pt.rename(columns=mapper)

    # median
    median_pt = pd.pivot_table(data, values='decision_id', index=["user_id"],
                               columns=['period'], aggfunc="median").reset_index()

    mapper = {1: "median_1", 2: "median_2", 3: "median_3", 4: "median_4"}

    median_pt = median_pt.rename(columns=mapper)

    # std
    std_pt = pd.pivot_table(data, values='decision_id', index=["user_id"],
                            columns=['period'], aggfunc="std").reset_index()

    mapper = {1: "std_1", 2: "std_2", 3: "std_3", 4: "std_4"}

    std_pt = std_pt.rename(columns=mapper)

    # sum
    sum_pt = pd.pivot_table(data, values='decision_id', index=["user_id"],
                            columns=['period'], aggfunc="sum").reset_index()

    mapper = {1: "sum_1", 2: "sum_2", 3: "sum_3", 4: "sum_4"}

    sum_pt = sum_pt.rename(columns=mapper)

    new_datas = [cnt_pt, max_pt, min_pt, mean_pt, median_pt, std_pt, sum_pt]
    return new_datas
