import os
import logging
import numpy as np
import pandas as pd

from rc.recommend import matrix_for_collab_filter, matrix_factorization, recommend_prod_by_user
from preprocessing.preprocessing import (standard_price, categorize_user, purchase_after_click, duplicated_click,
                                         weighted_popular_product)


def exists_user_recommendation():
    """
    pass
    :return:
    """
    # 필요 데이터 로드
    product_data = pd.read_csv(os.path.join(os.getcwd() + '/data' + '/_products.csv'))
    purchase_data = pd.read_csv(os.path.join(os.getcwd() + '/data' + '/_purchase_log.csv'))
    click_data = pd.read_csv(os.path.join(os.getcwd() + './data' + '/_click_log.csv'))
    test_data = pd.read_csv(os.path.join(os.getcwd() + './data' + '/_20230401_users.csv'))

    # 데이터 전처리
    std_p_data = standard_price(product_data)
    weighted_popular = weighted_popular_product(purchase_data, click_data)

    own_rating_products = pd.merge(std_p_data, weighted_popular, on='product_id')
    own_rating_products.fillna(0, inplace=True)
    own_rating_products['rating'] = own_rating_products['relative_price'] + own_rating_products['pop_rate']

    # purchase after click
    p_after_c = purchase_after_click(click_data, purchase_data)
    p_after_c = duplicated_click(p_after_c)

    user_prod_rating, prod_sim_df = matrix_for_collab_filter(purchase_data=p_after_c, product_data=own_rating_products)
    P, Q = matrix_factorization(user_prod_rating.values, K=50, steps=200, learning_rate=0.01, r_lambda=0.01)
    pred_matrix = np.dot(P, Q.T)
    ratings_pred_matrix = pd.DataFrame(data=pred_matrix,
                                       index=user_prod_rating.index,
                                       columns=user_prod_rating.columns)
    m_test_pur = pd.merge(test_data, p_after_c, on='user_id')
    exist_user = m_test_pur['user_id'].unique()

    result_dict = {'user_id': [], 'recommendation': []}
    for uid in exist_user:
        result_dict['user_id'].append(uid)
        result_dict['recommendation'].append(recommend_prod_by_user(ratings_pred_matrix,
                                                                    user_prod_rating, uid, top=1).index[0])
    return result_dict


def new_user_recommendation():
    pass

os.path.join(os.getcwd() + 'data' + '_purchase_log.csv')



