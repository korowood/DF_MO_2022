import pandas as pd


def create_place_and_score() -> pd.DataFrame:
    usr = pd.read_csv("user.csv")
    team_points = pd.read_csv("team_point.csv")

    team_points_usr = usr.merge(team_points, how="left", left_on="team_id", right_on="team_id")

    b = pd.pivot(team_points_usr[team_points_usr["period"].isin([0, 1, 2, 3, 4])],
                 index=["user_id"],
                 columns=['category_id', "period"],
                 values="score").reset_index()

    score_per_period = pd.DataFrame(data=b.drop("user_id", axis=1).values,
                                    columns=[f'{i}_per_{j}_score' for i in range(1, 7) for j in range(5)],
                                    index=b["user_id"])

    b = pd.pivot(team_points_usr[team_points_usr["period"].isin([0, 1, 2, 3, 4])],
                 index=["user_id"],
                 columns=['category_id', "period"],
                 values="place").reset_index()

    place_per_period = pd.DataFrame(data=b.drop("user_id", axis=1).values,
                                    columns=[f'{i}_per_{j}_place' for i in range(1, 7) for j in range(5)],
                                    index=b["user_id"])

    team_points_usr_gr = team_points_usr.groupby(["user_id", "game_id", "team_id"])["period"].apply(list).reset_index()
    team_points_usr_gr = team_points_usr_gr[["game_id", "team_id", "user_id"]].set_index("user_id")
    return pd.concat([place_per_period, score_per_period, team_points_usr_gr], axis=1).reset_index()
    # return team_points_usr_gr
