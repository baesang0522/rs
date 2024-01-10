import math
import pandas as pd


def standard_price(product_data):
    """
    각 prd id의 가격이 cat 내에서 평균에서 얼마만큼 떨어져 있는지 -> 가격을 대체할 것임
    :param product_data: pandas dataframe
    :return: changed
    """
    g_products = product_data.groupby('cat_idx')['price'].describe().reset_index()
    product_data = pd.merge(product_data, g_products[['cat_idx', 'mean']], on='cat_idx')
    product_data['relative_price'] = product_data.apply(lambda x:
                                                        math.log(math.exp((x['price'] - x['mean'])/x['mean']) + 1),
                                                        axis=1)
    return product_data


def categorize_user(user_data):
    """
    테스트셋에 기존 3월 데이터엔 없었던 신규 사용자에 대한 추천을 위해
    유저 카테고리화를 진행.
    기존 유저는 기존 데이터로 추천 진행

    :param user_data: train user data
    :return:
    """
    # unknown, nan 값 변경. 유저 카테고리화 위함
    user_data['age_range'].fillna('100', inplace=True)
    user_data['age_range'].replace('unknown', '100', inplace=True)
    user_data['gender'].fillna('3', inplace=True)
    user_data['gender'].replace('unknown', '3', inplace=True)

    # 카테고리화
    age_2_idx = {age: idx for idx, age in enumerate(sorted(user_data['age_range'].unique()))}
    gender_2_idx = {gender: idx for idx, gender in enumerate(sorted(user_data['gender'].unique()))}
    user_data['age_range'] = user_data['age_range'].map(age_2_idx)
    user_data['gender'] = user_data['gender'].map(gender_2_idx)
    return user_data








