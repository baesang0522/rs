import os
import logging
import numpy as np
import pandas as pd


from sklearn.utils import shuffle

from config import config as c
from rc.recommendation import MatrixFactorization
from preprocessing.preprocessing import (standard_price, categorize_user, purchase_after_click, duplicated_click,
                                         weighted_popular_product, to_matrix)

config = c['DEV']


def new_user_preprocess(user_data, click_data, purchase_data):
    remain_col_list, rename_col_dict = ['user_id_y', 'product_id', 'measure', 'dt'], {'user_id_y': 'user_id'}

    cat_user_click = pd.merge(click_data, user_data, left_on='user_id', right_on='old')
    cat_user_pur = pd.merge(purchase_data, user_data, left_on='user_id', right_on='old')

    cat_user_click = cat_user_click[remain_col_list]
    cat_user_pur = cat_user_pur[remain_col_list]

    cat_user_click.rename(columns=rename_col_dict, inplace=True)
    cat_user_pur.rename(columns=rename_col_dict, inplace=True)
    return cat_user_click, cat_user_pur


def train_process(user_prod_rating, exist_or_not):
    ratings = shuffle(user_prod_rating, random_state=2021)
    cutoff = int(config.TRAIN_SIZE * len(ratings))
    ratings_test = ratings.iloc[cutoff:]
    if exist_or_not == 'exists':
        mf = MatrixFactorization(ratings, config.EXIST_HYPER_PARAMETERS)
    else:
        mf = MatrixFactorization(ratings, config.NOT_EXIST_HYPER_PARAMETERS)
    mf.set_test(ratings_test)
    mf.test()
    return mf


def exists_user_recommendation(product_data, purchase_data, click_data, test_data, topN=1):
    # 데이터 전처리
    weighted_popular = weighted_popular_product(purchase_data, click_data)

    own_rating_products = pd.merge(product_data, weighted_popular, on='product_id')
    own_rating_products.fillna(0, inplace=True)
    own_rating_products['rating'] = own_rating_products['relative_price'] + own_rating_products['pop_rate']
    
    # purchase after click
    p_after_c = purchase_after_click(click_data, purchase_data)
    p_after_c = duplicated_click(p_after_c)

    user_prod_rating = to_matrix(c_or_p_data=p_after_c, product_data=own_rating_products, cp_type='purchase')
    mf = train_process(user_prod_rating, exist_or_not='exists')
    m_test_pur = pd.merge(test_data, p_after_c, on='user_id')
    exist_user = m_test_pur['user_id'].unique()

    result_dict = {'user_id': [], 'recommendation': []}
    for uid in exist_user:
        result_dict['user_id'].append(uid)
        result_dict['recommendation'].append(mf.recommend_prod_by_user(user_prod_rating, user_id=uid).index[topN])
    return result_dict, exist_user


def new_user_recommendation(user_data, product_data, purchase_data, click_data, test_data, exist_user_list, topN=1):
    test_data = test_data[~test_data['user_id'].isin(exist_user_list)]
    user_data, to_predict_user_data = categorize_user(user_data, test_data)

    cat_user_click, cat_user_pur = new_user_preprocess(user_data, click_data, purchase_data)
    weighted_popular = weighted_popular_product(cat_user_pur, cat_user_click)

    own_rating_products = pd.merge(product_data, weighted_popular, on='product_id')
    own_rating_products.fillna(0, inplace=True)
    own_rating_products['rating'] = own_rating_products['relative_price'] + own_rating_products['pop_rate']

    # 실 구매에선 없는 카테고리도 있음. 해당 카테고리의 유저들에겐 단순 클릭 데이터로 추천을 해 줄 것임.
    p_after_c = purchase_after_click(cat_user_click, cat_user_pur)
    p_after_c = duplicated_click(p_after_c)

    user_prod_rating = to_matrix(c_or_p_data=p_after_c, product_data=own_rating_products, cp_type='purchase')
    mf = train_process(user_prod_rating, exist_or_not='not_exist')

    not_in_cat = [cat_id for cat_id in user_data['user_id'].unique() if cat_id not in p_after_c['user_id'].unique()]

    no_exists_1 = to_predict_user_data[~to_predict_user_data['user_id'].isin(not_in_cat)]
    exist_user = no_exists_1['user_id'].unique()
    result_dict = {'user_id': [], 'recommendation': []}
    for uid in exist_user:
        result_dict['user_id'].append(uid)
        result_dict['recommendation'].append(mf.recommend_prod_by_user(user_prod_rating, user_id=uid).index[topN])

    user_prod_rating = to_matrix(c_or_p_data=cat_user_click, product_data=own_rating_products, cp_type='click')
    print(cat_user_click.columns)
    print(2222222222222222222222222222222222222222222222)
    mf = train_process(user_prod_rating, exist_or_not='not_exist')
    for uid in not_in_cat:
        result_dict['user_id'].append(uid)
        result_dict['recommendation'].append(mf.recommend_prod_by_user(user_prod_rating, user_id=uid).index[topN])
    cat_df = pd.DataFrame(result_dict)
    result_df = pd.merge(to_predict_user_data, cat_df, on='user_id')
    result_df = result_df[['old', 'recommendation']]
    result_df = result_df.rename(columns={'old': 'user_id'})
    return dict(zip(result_df.user_id, result_df.recommendation))


def main_functions(user_data_name, product_data_name, purchase_data_name, click_data_name, test_data_name):
    user_data = pd.read_csv(os.path.join(config.DATA_PATH, user_data_name))
    product_data = pd.read_csv(os.path.join(config.DATA_PATH, product_data_name))
    purchase_data = pd.read_csv(os.path.join(config.DATA_PATH, purchase_data_name))
    click_data = pd.read_csv(os.path.join(config.DATA_PATH, click_data_name))
    test_data = pd.read_csv(os.path.join(config.DATA_PATH, test_data_name))
    std_p_data = standard_price(product_data)

    exists_dict, exist_user_list = exists_user_recommendation(std_p_data, purchase_data, click_data,
                                                              test_data, topN=1)
    no_exist_dict = new_user_recommendation(user_data, std_p_data, purchase_data, click_data, test_data,
                                            exist_user_list, topN=1)
    exists_dict['user_id'].expand(no_exist_dict['user_id'])
    exists_dict['recommendation'].expand(no_exist_dict['recommendation'])
    exists_df = pd.DataFrame(exists_dict)
    result_df = pd.merge(test_data, exists_df, on='user_id')
    result_df.to_csv(os.path.join(config.DATA_PATH, 'result.csv'))
    




#
#
#
#
#
#
# os.path.join(os.getcwd() + 'data' + '_purchase_log.csv')
# len(p_after_c['user_id'].unique()) # 11
# len(cat_user_click['user_id'].unique()) # 14
# len(user_data['user_id'].unique()) # 14
# len(to_predict_user_data['user_id'].unique()) # 13
#
#
# for id in user_data['user_id'].unique():
#     if id not in p_after_c['user_id'].unique():
#         print(id)
#
#
# aa = 'abcd'
# if aa[1] == 'b':
#     aa[1] = 'r'
#
# dd = {'a': [1,2,3,4,5], 'b': [2,3,4,5,6]}
# pd.DataFrame(dd).to_dict(orient='records')
