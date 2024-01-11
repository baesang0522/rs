import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def prd_category_to_idx(category_list):
    return {category: idx for idx, category in enumerate(set(category_list))}


def top_k_items(item_id, top_k, corr_mat, map_name):
    # sort correlation value ascendingly and select top_k item_id
    top_items = corr_mat[item_id, :].argsort()[-top_k:][::-1]
    top_items = [map_name[e] for e in top_items]
    return top_items


def get_prod_base_collab(collab_data, idx):
    return prod_based_collab[args].sort_values(ascending=False)[:6]


def cos_sim_mat(product_data, purchase_data):
    pur_pro = product_data.loc[product_data['product_id'].isin(purchase_data['product_id'])].copy()
    category = pur_pro['category'].str.split(",", expand=True)

    all_cat = set()
    for c in category.columns:
        distinct_genre = category[c].str.lower().str.strip().unique()
        all_cat.update(distinct_genre)
    all_cat.remove(None)
    # create item-genre matrix
    item_cat_mat = pur_pro[['product_id', 'category']].copy()
    item_cat_mat['category'] = item_cat_mat['category'].str.lower().str.strip()

    # OHE the genres column
    item_cat_mat = pd.concat([item_cat_mat, item_cat_mat['category'].str.get_dummies(sep=',')], axis=1)
    item_cat_mat = item_cat_mat.drop(['category'], axis=1)
    item_cat_mat = item_cat_mat.set_index('product_id')
    corr_mat = cosine_similarity(item_cat_mat)
    return corr_mat


def matrix_for_collab_filter(purchase_data, product_data):
    g_purchase_cnt = purchase_data[['user_id', 'product_id']].groupby(['product_id'], as_index=False).count()
    g_purchase_cnt.rename(columns={'product_id': 'product_id', 'user_id': 'cnt'}, inplace=True)
    g_purchase_cnt['cnt'] = (g_purchase_cnt['cnt'] / len(purchase_data))
    g_purchase_sum = purchase_data[['user_id', 'product_id', 'measure_purchase']].groupby(['user_id', 'product_id'],
                                                                                 as_index=False).sum()
    g_purchase = pd.merge(g_purchase_sum, g_purchase_cnt, on=['product_id'])
    g_purchase['rating'] = g_purchase['measure_purchase'] * g_purchase['cnt']
    g_pp = pd.merge(product_data, g_purchase, on=['product_id'], how='left', suffixes=('_prod', '_pur'))
    g_pp.fillna(0, inplace=True)
    g_pp['rating'] = g_pp['rating_prod'] + 1 + g_pp['rating_pur']
    # g_pp = pd.merge(g_purchase, product_data, on=['product_id'])

    user_prod_rating = g_pp.pivot_table('rating', index='user_id', columns='product_id')
    user_prod_rating.fillna(0, inplace=True)
    prod_user_rating = user_prod_rating.T

    prod_sim = cosine_similarity(prod_user_rating, prod_user_rating)
    prod_sim_df = pd.DataFrame(prod_sim, index=prod_user_rating.index, columns=prod_user_rating.index)
    return user_prod_rating, prod_sim_df


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


def get_rmse(R, P, Q, non_zeros):
    error = 0
    full_pred_matrix = np.dot(P, Q.T)
    x_non_zero = [non_zero[0] for non_zero in non_zeros]
    y_non_zero = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero, y_non_zero]

    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero, y_non_zero]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    return np.sqrt(mse)


def predict_rating_top(rating_arr, prod_sim_arr, N=5):
    pred = np.zeros(rating_arr.shape)

    for col in range(rating_arr.shape[1]):
        temp = np.argsort(prod_sim_arr[:, col])
        top_n_items = [temp[:-1-N:-1]]
        for row in range(rating_arr.shape[0]):
            prod_sim_arr_topN = prod_sim_arr[col, :][top_n_items].T
            ratings_arr_topN = rating_arr[row, :][top_n_items]

            pred[row, col] = ratings_arr_topN @ prod_sim_arr_topN
            pred[row, col] /= np.sum(np.abs(prod_sim_arr_topN))
    return pred


def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    break_count = 0

    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    for step in range(steps):
        for i, j, r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :].T)
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])
        rmse = get_rmse(R, P, Q, non_zeros)
        if step % 10 == 0:
            print("### iteration step : ", step," rmse : ", rmse)
    return P, Q


def recommend_prod_by_user(pred_df, rating_matrix, user_id, top=5):
    user_rating = rating_matrix.loc[user_id, :]
    bought = user_rating[user_rating > 0].index.tolist()

    prod_list = rating_matrix.columns.tolist()
    unseen_list = [movie for movie in prod_list if movie not in bought]

    recomm_prod = pred_df.loc[user_id, unseen_list].sort_values(ascending=False)[:top]
    recomm_prod_df = pd.DataFrame(data=recomm_prod.values,
                                    index=recomm_prod.index,
                                    columns=['pred_score'])

    return recomm_prod_df


ratings_pred = predict_rating_top(user_prod_rating.values, prod_sim_df.values, N=5)
get_mse(ratings_pred, user_prod_rating.values)

P, Q = matrix_factorization(user_prod_rating.values, K=50, steps=200, learning_rate=0.01, r_lambda=0.01)
pred_matrix = np.dot(P, Q.T)
ratings_pred_matrix = pd.DataFrame(data=pred_matrix,
                                   index=user_prod_rating.index,
                                   columns=user_prod_rating.columns)

recommend_prod_by_user(ratings_pred_matrix, user_prod_rating, 306, top=10)











# def predict_rating(rating_arr, item_sim_arr):
#     sum_sr = rating_arr @ item_sim_arr
#     sum_s_abs = np.array([np.abs(item_sim_arr).sum(axis=1)])
#     rating_pred = sum_sr / sum_s_abs
#     return rating_pred
#
# rating_pred = predict_rating(user_prod_rating, prod_sim_df)
# rating_pred_matrix = pd.DataFrame(data=rating_pred, index= user_prod_rating.index,
#                                    columns = user_prod_rating.columns)
