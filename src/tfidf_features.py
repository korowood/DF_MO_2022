import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 200


def create_tfidf_features():
    df_user = pd.read_csv("./data/user_decision.csv")
    df_user_train_gr = df_user.groupby(["user_id", "period"])["decision_id"].apply(list).reset_index(name='new')

    vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=MAX_FEATURES)
    tmp = df_user_train_gr['new'].apply(lambda x: ' '.join(str(y) for y in x)).values

    X_vec = vectorizer.fit_transform(tmp).toarray()
    pd_vec = pd.DataFrame(data=X_vec, columns=[f'vec_{i}' for i in range(MAX_FEATURES)])
    pd_vec['user_id'] = df_user_train_gr['user_id']
    pd_vec['period'] = df_user_train_gr['period']

    return pd.pivot_table(pd_vec[pd_vec["period"].isin([0, 1, 2, 3, 4])],
                          index=["user_id"],
                          values=[f'vec_{i}' for i in range(MAX_FEATURES)], )

    # a = pd_vec[pd_vec['period'] == 1].rename(columns={f'vec_{i}': f'1_vec_{i}' for i in range(MAX_FEATURES)})
    #
    # b = pd_vec[pd_vec['period'] == 2].rename(columns={f'vec_{i}': f'2_vec_{i}' for i in range(MAX_FEATURES)})
    #
    # c = pd_vec[pd_vec['period'] == 3].rename(columns={f'vec_{i}': f'3_vec_{i}' for i in range(MAX_FEATURES)})
    #
    # d = pd_vec[pd_vec['period'] == 4].rename(columns={f'vec_{i}': f'4_vec_{i}' for i in range(MAX_FEATURES)})
    #
    # return a.reset_index(drop=True), b.reset_index(drop=True), c.reset_index(drop=True), d.reset_index(drop=True)


#%%
