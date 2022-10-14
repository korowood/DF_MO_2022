import pandas as pd
from .dicision_stats_features import create_decision


PATH_SOLUTION = "./data/sample_solution.csv"
PATH_USER = "./data/user_decision.csv"
PATH_TFIDF = "./artifacts/tfidf_features.csv"
PATH_SCORE = "./artifacts/place_score.csv"
PATH_IMPUTER = "./artifacts/imp_mean.pkl"

PATH_RES = "./artifacts/test_vec_scores_nans.csv"


def calc_result() -> pd.DataFrame:
    subs = pd.read_csv(PATH_SOLUTION)
    df_user = pd.read_csv(PATH_USER)

    list_id = subs["id"].unique().tolist()

    df_user_test = df_user[df_user["user_id"].isin(list_id)]

    # get stats metrics
    new_datas = create_decision(df_user_test)
    test = subs.copy()
    for data in new_datas:
        test = test.merge(data, how="left", left_on="id", right_on="user_id").drop("user_id", axis=1)

    # get embeddings
    tfidf_features = pd.read_csv(PATH_TFIDF, index_col='user_id')
    test_w_vec = test.merge(tfidf_features.reset_index(), how="left", left_on="id", right_on="user_id").drop("user_id",
                                                                                                             axis=1)
    # get score
    place_score = pd.read_csv(PATH_SCORE)
    test_vec_scores = test_w_vec.merge(place_score, how="left", left_on="id", right_on="user_id")

    # replace NaN
    imp_mean = pd.read_pickle("./artifacts/imp_mean.pkl")
    tr_data = imp_mean.transform(test_vec_scores)

    test_vec_scores_nans = pd.DataFrame(tr_data, columns=test_vec_scores.columns)
    test_vec_scores_nans.to_csv(PATH_RES)

    return test_vec_scores_nans


def get_ans(*args):
    subs = pd.read_csv(PATH_SOLUTION)
    ans = subs.copy()

    ans["Adaptability"] = args[0]
    ans["Systemic thinking"] = args[1]
    ans["Analytical thinking"] = args[2]
    ans["Focus"] = args[3]

    save_file_name = input()
    ans.to_csv(f"{save_file_name}.csv", index=False)

    return ans
