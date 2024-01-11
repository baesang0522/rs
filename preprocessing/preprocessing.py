import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prd_category_to_idx(category_list):
    return {category: idx for idx, category in enumerate(set(category_list))}


def standard_price(product_data):
    """
    각 prd id의 가격이 cat 내에서 평균에서 얼마만큼 떨어져 있는지 -> 가격을 대체할 것임
    역수 취해줌. 카테고리 내에서 가격이 상대적으로 싸면 점수가 높게

    :param product_data: pandas dataframe
    :return: changed
    """
    # product_data['cat_idx'] = product_data['category'].map(prd_category_to_idx(product_data['category']))
    g_products = product_data.groupby('category')['price'].describe().reset_index()
    product_data = pd.merge(product_data, g_products[['category', 'mean']], on='category')
    product_data['relative_price'] = product_data.apply(lambda x:
                                                        1/(math.log(math.exp((x['price'] - x['mean']) / x['mean']) + 1)),
                                                        axis=1)
    return product_data


def user_age_recat(user_data):
    og_age_range = sorted(user_data['age_range'].astype(str).unique())[:-2]
    # 앞에 둘, 뒤에 둘 나이 서로 묶음((-14, 15-19), (60-64, 65-69, 70-))
    del_age_cat = og_age_range[:2] + og_age_range[-3:]
    user_data['age_range'].fillna('100', inplace=True)
    user_data['age_range'].replace('unknown', '100', inplace=True)

    for age in del_age_cat[:2]:
        user_data['age_range'].replace(age, '-19', inplace=True)
    for age in del_age_cat[2:]:
        user_data['age_range'].replace(age, '60-', inplace=True)

    user_data['age_range'] = user_data['age_range'].apply(lambda x: x[0] + '0' + x[2:] if x[1] == '5' and len(x) > 3
                                                                                    else x[:-1] + '9')
    return user_data




def categorize_user(user_data, to_predict_user_data):
    """
    테스트셋에 기존 3월 데이터엔 없었던 신규 사용자에 대한 추천을 위한 categorize.
    유저 카테고리화를 진행.
    기존 유저는 기존 데이터로 추천 진행

    :param user_data: 3월 user data
    :param to_predict_user_data: 4월 user data
    :return: user_data, to_predict_user_data
    """
    # 카테고리화. 실 구매에서는 없는 카테고리 존재. 현재는 나이 범위를 좀 늘릴 필요가 있음.
    # 20대 초, 20대 후 -> 20대
    # unknown, nan 값 변경. 유저 카테고리화 위함
    user_data = user_age_recat(user_data)
    user_data['gender'].fillna('3', inplace=True)
    user_data['gender'].replace('unknown', '3', inplace=True)

    to_predict_user_data = user_age_recat(to_predict_user_data)
    to_predict_user_data['gender'].fillna('3', inplace=True)
    to_predict_user_data['gender'].replace('unknown', '3', inplace=True)

    age_2_idx = {age: idx for idx, age in enumerate(sorted(user_data['age_range'].unique()))}
    gender_2_idx = {gender: idx for idx, gender in enumerate(sorted(user_data['gender'].unique()))}
    user_data['age_range'] = user_data['age_range'].map(age_2_idx)
    user_data['gender'] = user_data['gender'].map(gender_2_idx)

    to_predict_user_data['age_range'] = to_predict_user_data['age_range'].map(age_2_idx)
    to_predict_user_data['gender'] = to_predict_user_data['gender'].map(gender_2_idx)

    user_data['cat'] = user_data.apply(lambda x: str(x['age_range']) + str(x['gender']), axis=1)
    to_predict_user_data['cat'] = to_predict_user_data.apply(lambda x: str(x['age_range']) + str(x['gender']), axis=1)

    user_data.rename(columns={'user_id': 'old', 'cat': 'user_id'}, inplace=True)
    to_predict_user_data.rename(columns={'user_id': 'old', 'cat': 'user_id'}, inplace=True)

    return user_data, to_predict_user_data


# purchase data 와 click data 를 엮어 클릭해서 산 user만 추출함. click dt <= purchase dt 인 조건
def purchase_after_click(click_data, purchase_data):
    p_after_c = pd.merge(click_data, purchase_data, on=['user_id', 'product_id'], suffixes=('_click', '_purchase'))
    p_after_c = p_after_c.where(p_after_c['dt_click'] <= p_after_c['dt_purchase'])
    p_after_c = p_after_c.sort_values(by=['user_id', 'product_id', 'dt_click', 'dt_purchase'],
                                      ascending=True).reset_index(drop=True)
    p_after_c.drop_duplicates(['user_id', 'product_id', 'dt_click'], inplace=True)
    return p_after_c


# 여러날에 걸쳐 여러번 클릭해보다 한번 산 경우 반영 필요. 중복됨.
def duplicated_click(p_after_c):
    condition_1 = [['user_id', 'product_id', 'dt_purchase'], ['user_id', 'product_id', 'dt_purchase', 'measure_click']]
    condition_2 = [['user_id', 'product_id', 'dt_click', 'dt_purchase'],
                   ['user_id', 'product_id', 'dt_purchase', 'measure_purchase']]

    p_after_c_1 = p_after_c.groupby(condition_1[0], as_index=False).sum()
    p_after_c_2 = p_after_c.groupby(condition_2[0], as_index=False).sum()
    p_after_c = pd.merge(p_after_c_1[condition_1[1]], p_after_c_2[condition_2[1]],
                         on=['user_id', 'product_id', 'dt_purchase'])
    p_after_c.drop_duplicates(['user_id', 'product_id', 'dt_purchase', 'measure_click', 'measure_purchase'],
                              inplace=True)
    return p_after_c


# 단순 인기순. 많은 사람들이 구매한 순으로 결정함. 이후 성별 구매 많은 순 데이터 뽑으면 될듯?
def weighted_popular_product(purchase_data, click_data):
    scaler = MinMaxScaler()

    popular_desc = purchase_data.groupby(['product_id'], as_index=False).count()
    popular_desc.sort_values(by='measure', ascending=False, inplace=True)
    popular_desc.reset_index(drop=True, inplace=True)
    popular_desc['tot_pur'] = len(popular_desc)
    popular_desc = popular_desc[['product_id', 'user_id', 'tot_pur']]
    popular_desc.rename(columns={'product_id': 'product_id', 'user_id': 'purchase_cnt',
                                 'tot_pur': 'tot_pur'}, inplace=True)
    popular_desc['pur_rate'] = popular_desc.apply(lambda x: x['purchase_cnt'] / x['tot_pur'] * 10, axis=1)

    # click to purchase. based on prd_id
    click_desc = click_data[['product_id', 'measure']].groupby(['product_id'], as_index=False).sum()
    click_desc.sort_values(by='measure', ascending=False, inplace=True)
    click_desc.reset_index(drop=True, inplace=True)
    click_desc['tot_click'] = len(click_desc)
    click_desc['click_rate'] = click_desc.apply(lambda x: x['measure'] / x['tot_click'] * 1000, axis=1)

    wpp = pd.merge(click_desc, popular_desc, on='product_id', how='left').fillna(0)
    wpp['pop_rate'] = wpp.apply(lambda x: (x['purchase_cnt'] / x['measure']) * x['click_rate'] * x['pur_rate'], axis=1)
    wpp.sort_values('pop_rate', ascending=False, inplace=True)
    wpp[['pop_rate']] = scaler.fit_transform(wpp[['pop_rate']])
    wpp['pop_rate'] = wpp['pop_rate'].apply(lambda x: math.log(x+1))
    wpp = wpp[['product_id', 'pop_rate']]
    wpp = wpp.reset_index(drop=True)
    return wpp.groupby('product_id', as_index=False).sum()



