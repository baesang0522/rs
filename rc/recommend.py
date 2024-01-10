import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
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


def collab_filter(purchase_data, product_data):
    g_purchase_cnt = purchase_data[['user_id', 'product_id']].groupby(['product_id'], as_index=False).count()
    g_purchase_cnt.rename(columns={'product_id': 'product_id', 'user_id': 'cnt'}, inplace=True)
    g_purchase_cnt['cnt'] = (g_purchase_cnt['cnt'] / len(purchase_data)) * 10
    g_purchase_sum = purchase_data[['user_id', 'product_id', 'measure']].groupby(['user_id', 'product_id'],
                                                                                 as_index=False).sum()
    g_purchase = pd.merge(g_purchase_sum, g_purchase_cnt, on=['product_id'])
    g_purchase['rating'] = g_purchase['measure'] * g_purchase['cnt']
    g_pp = pd.merge(g_purchase, product_data, on=['product_id'])

    user_prod_rating = g_pp.pivot_table('rating', index='user_id', columns='product_id')
    user_prod_rating.fillna(0, inplace=True)
    prod_user_rating = user_prod_rating.T

    prod_sim = cosine_similarity(prod_user_rating, prod_user_rating)
    prod_sim_df = pd.DataFrame(prod_sim, index=prod_user_rating.index, columns=prod_user_rating.index)


def predict_rating(rating_arr, item_sim_arr):
    sum_sr = rating_arr @ item_sim_arr
    sum_s_abs = np.array([np.abs(item_sim_arr).sum(axis=1)])
    rating_pred = sum_sr / sum_s_abs
    return rating_pred

rating_pred = predict_rating(user_prod_rating, prod_sim_df)
rating_pred_matrix = pd.DataFrame(data=rating_pred, index= user_prod_rating.index,
                                   columns = user_prod_rating.columns)

from sklearn.metrics import mean_squared_error


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


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

ratings_pred = predict_rating_top(user_prod_rating.values, prod_sim_df.values, N=5)
get_mse(ratings_pred, user_prod_rating.values)

