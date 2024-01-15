import os
import pandas as pd

from collections import defaultdict
from sklearn.utils import shuffle

from config import config as c
from rc.recommendation import MatrixFactorization
from preprocessing.preprocessing import (standard_price, categorize_user, preprocessing_pur_click_data,
                                         new_user_preprocess, making_matrix)

config = c['DEV']


def train_process(user_prod_rating, exist_or_not):
    """
    train process wrapping 함수
    :param user_prod_rating: train data
    :param exist_or_not: 구매이력 & test에 존재하는 user 인지 아닌지
    :return: 학습 모델 객체
    """
    ratings = shuffle(user_prod_rating, random_state=2021)
    cutoff = int(config.TRAIN_SIZE * len(ratings))
    ratings_test = ratings.iloc[cutoff:].stack().reset_index()
    if exist_or_not == 'exists':
        mf = MatrixFactorization(ratings, config.EXIST_HYPER_PARAMETERS)
    else:
        mf = MatrixFactorization(ratings, config.NOT_EXIST_HYPER_PARAMETERS)
    mf.set_test(ratings_test)
    mf.train()
    return mf


def exists_user_recommendation(product_data, purchase_data, click_data, test_data):
    """
    구매이력 & test에 존재하는 user 에 대한 추천모델 학습 & 예측 함수
    :param product_data: product_csv
    :param purchase_data: purchase_csv
    :param click_data: click_csv
    :param test_data: _xxxx_users_csv
    :return: {'user_id': [product1, product2, ..., product5], ...}, 구매이력 & test에 존재하는 user id 리스트
    """
    p_after_c, own_rating_products = preprocessing_pur_click_data(purchase_data, click_data, product_data)

    user_prod_rating = making_matrix(c_or_p_data=p_after_c, product_data=own_rating_products, cp_type='purchase')
    print('#########################################')
    print("start exists user model training...")
    print('#########################################')
    mf = train_process(user_prod_rating, exist_or_not='exists')
    m_test_pur = pd.merge(test_data, p_after_c, on='user_id')
    exist_user = m_test_pur['user_id'].unique()

    result_dict = defaultdict(list)
    for uid in exist_user:
        result_dict[uid] = list(mf.recommend_prod_by_user(user_prod_rating, own_rating_products, user_id=uid).index)
    print('#########################################')
    print("train exists user model finished...")
    print('#########################################')
    return result_dict, exist_user


def new_user_recommendation(user_data, product_data, purchase_data, click_data, test_data, exist_user_list):
    """
    구매이력 & test에 존재하는 user 에 대한 추천모델 학습 & 예측 함수
    :param user_data: user_csv
    :param product_data: product_csv
    :param purchase_data: purchase_csv
    :param click_data: click_csv
    :param test_data: _xxxx_users_csv
    :param exist_user_list: 구매이력 & test에 존재하는 user id 리스트
    :return: {'user_id': [product1, product2, ..., product5], ...}
    """
    test_data = test_data[~test_data['user_id'].isin(exist_user_list)]
    user_data, to_predict_user_data = categorize_user(user_data, test_data)

    cat_user_click, cat_user_pur = new_user_preprocess(user_data, click_data, purchase_data)
    p_after_c, own_rating_products = preprocessing_pur_click_data(cat_user_pur, cat_user_click, product_data)

    user_prod_rating = making_matrix(c_or_p_data=p_after_c, product_data=own_rating_products, cp_type='purchase')
    print('#########################################')
    print("start not exists user model training...")
    print('#########################################')
    mf = train_process(user_prod_rating, exist_or_not='not_exist')

    not_in_cat = [cat_id for cat_id in user_data['user_id'].unique() if cat_id not in p_after_c['user_id'].unique()]

    no_exists_1 = to_predict_user_data[~to_predict_user_data['user_id'].isin(not_in_cat)]
    exist_user = no_exists_1['user_id'].unique()

    result_dict = defaultdict(list)
    for uid in exist_user:
        result_dict[uid] = list(mf.recommend_prod_by_user(user_prod_rating, own_rating_products, user_id=uid).index)

    user_prod_rating = making_matrix(c_or_p_data=cat_user_click, product_data=own_rating_products, cp_type='click')
    mf = train_process(user_prod_rating, exist_or_not='not_exist')
    for uid in not_in_cat:
        result_dict[uid] = list(mf.recommend_prod_by_user(user_prod_rating, own_rating_products, user_id=uid).index)
    cat_df = pd.DataFrame({'user_id': list(result_dict.keys()), 'recommendation': list(result_dict.values())})
    result_df = pd.merge(to_predict_user_data, cat_df, on='user_id')
    result_df = result_df[['old', 'recommendation']]
    result_df = result_df.rename(columns={'old': 'user_id'})
    print('#########################################')
    print("train not exists user model finished...")
    print('#########################################')
    return {row.user_id: row.recommendation for row in result_df.itertuples()}


def main_functions(user_data_name, product_data_name, purchase_data_name, click_data_name, test_data_name):
    """
    추천 모델 train wrapping 함수(exists & new)

    :param user_data_name: user_csv file name
    :param product_data_name: product_csv file name
    :param purchase_data_name: purchase_csv file name
    :param click_data_name: click_csv file name
    :param test_data_name: _xxxx_users_csv file name
    :return:
    """
    user_data = pd.read_csv(os.path.join(config.DATA_PATH, user_data_name))
    product_data = pd.read_csv(os.path.join(config.DATA_PATH, product_data_name))
    purchase_data = pd.read_csv(os.path.join(config.DATA_PATH, purchase_data_name))
    click_data = pd.read_csv(os.path.join(config.DATA_PATH, click_data_name))
    test_data = pd.read_csv(os.path.join(config.DATA_PATH, test_data_name))

    # 이상치 우선 대략적으로 삭제함. 나중에 수정
    purchase_data = purchase_data[purchase_data['measure'] < 10]
    click_data = click_data[click_data['measure'] < 100]
    std_p_data = standard_price(product_data)

    exists_dict, exist_user_list = exists_user_recommendation(std_p_data, purchase_data, click_data,
                                                              test_data)
    no_exist_dict = new_user_recommendation(user_data, std_p_data, purchase_data, click_data, test_data,
                                            exist_user_list)

    exists_dict.update(no_exist_dict)
    result_df = pd.DataFrame({'user_id': list(exists_dict.keys()), 'recommendation': list(exists_dict.values())})
    result_df = pd.merge(test_data, result_df, on='user_id')
    result_df.to_csv(os.path.join(config.DATA_PATH, 'result.csv'), index=False)
    

