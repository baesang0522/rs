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
    epsilon, n_latent_factors = 1e-9, 10

    g_purchase_cnt = purchase_data[['user_id', 'product_id']].groupby(['product_id'], as_index=False).count()
    g_purchase_cnt.rename(columns={'product_id': 'product_id', 'user_id': 'cnt'}, inplace=True)
    g_purchase_cnt['cnt'] = (g_purchase_cnt['cnt'] / len(purchase_data)) * 10
    g_purchase_sum = purchase_data[['user_id', 'product_id', 'measure']].groupby(['user_id', 'product_id'],
                                                                                 as_index=False).sum()
    g_purchase = pd.merge(g_purchase_sum, g_purchase_cnt, on=['product_id'])
    g_purchase['rating'] = g_purchase['measure'] * g_purchase['cnt']
    g_pp = pd.merge(g_purchase, product_data, on=['product_id'])

    prod_user_rating = g_pp.pivot_table('rating', index='product_id', columns='user_id')
    user_prod_rating = g_pp.pivot_table('rating', index='user_id', columns='product_id')

    prod_user_rating.fillna(0, inplace=True)
    prod_based_collab = cosine_similarity(prod_user_rating)

    prod_based_collab = pd.DataFrame(data=prod_based_collab, index=prod_user_rating.index,
                                     columns=prod_user_rating.index)



# calculate sparsity
sparsity = float(len(mat.nonzero()[0]))
sparsity /= (mat.shape[0] * mat.shape[1])
sparsity *= 100
print(f'Sparsity: {sparsity:4.5f}%. This means that {sparsity:4.5f}% of the user-item ratings have a value.')

# compute similarity
item_corr_mat = cosine_similarity(mat.T)

# get top k item
print("\nThe top-k similar movie to item_id 99")
ind2name = {ind:name for ind,name in enumerate(item_genre_mat.index)}
name2ind = {v:k for k,v in ind2name.items()}
similar_items = top_k_items(name2ind['99'],
                            top_k = 10,
                            corr_mat = item_corr_mat,
                            map_name = ind2name)

display(items.loc[items[ITEM_COL].isin(similar_items)])