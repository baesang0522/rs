import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standard_price(product_data):
    """
    각 prd id의 가격이 cat 내에서 평균에서 얼마만큼 떨어져 있는지 -> 가격을 대체할 것임
    역수 취해줌. 카테고리 내에서 가격이 상대적으로 싸면 점수가 높게

    :param product_data: pandas dataframe
    :return: changed
    """
    g_products = product_data.groupby('category')['price'].describe().reset_index()
    product_data = pd.merge(product_data, g_products[['category', 'mean']], on='category')
    product_data['relative_price'] = product_data.apply(lambda x:
                                                        1/(math.log(math.exp((x['price'] - x['mean']) / x['mean']) + 1)),
                                                        axis=1)
    return product_data


def user_age_recat(user_data):
    """
    유저 age_range 카테고리화 함수
    :param user_data: user_data
    :return: categorized age_range user_data
    """
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

    카테고리화. 실 구매에서는 없는 카테고리 존재. 현재는 나이 범위를 좀 늘릴 필요가 있음.
    20대 초, 20대 후 -> 20대
    unknown, nan 값 변경. 유저 카테고리화 위함

    :param user_data: 3월 user data
    :param to_predict_user_data: 4월 user data
    :return: user_data, to_predict_user_data
    """
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


def purchase_after_click(click_data, purchase_data):
    """
    purchase data 와 click data 를 엮어 클릭해서 산 user만 추출함.
    click dt <= purchase dt 인 조건

    :param click_data: click_data
    :param purchase_data: purchase_data
    :return: merged click, purchased data.
    """
    p_after_c = pd.merge(click_data, purchase_data, on=['user_id', 'product_id'], suffixes=('_click', '_purchase'))
    p_after_c = p_after_c.where(p_after_c['dt_click'] <= p_after_c['dt_purchase'])
    p_after_c = p_after_c.sort_values(by=['user_id', 'product_id', 'dt_click', 'dt_purchase'],
                                      ascending=True).reset_index(drop=True)
    p_after_c.drop_duplicates(['user_id', 'product_id', 'dt_click'], inplace=True)
    return p_after_c


def duplicated_click(p_after_c):
    """
    여러날에 걸쳐 여러번 클릭해보다 한번 산 경우 반영 필요. 중복됨.

    :param p_after_c: purchased_after_click return 값
    :return: 중복이 제거된 p_after_c
    """
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


def making_matrix(c_or_p_data, product_data, cp_type):
    """
     mf train set 형태 제작 함수

    :param c_or_p_data: click_data or purchase_data
    :param product_data: product_data
    :param cp_type: click or purchase
    :return: pivoted pd.DataFrame
    """
    g_purchase_cnt = c_or_p_data[['user_id', 'product_id']].groupby(['product_id'], as_index=False).count()
    g_purchase_cnt.rename(columns={'product_id': 'product_id', 'user_id': 'cnt'}, inplace=True)
    g_purchase_cnt['cnt'] = (g_purchase_cnt['cnt'] / len(c_or_p_data))

    if cp_type == 'click':
        g_purchase_sum = c_or_p_data[['user_id', 'product_id', 'measure']].groupby(['user_id', 'product_id'],
                                                                                   as_index=False).sum()
    else:
        g_purchase_sum = c_or_p_data[['user_id', 'product_id', 'measure_purchase']].groupby(['user_id', 'product_id'],
                                                                                            as_index=False).sum()
    g_purchase = pd.merge(g_purchase_sum, g_purchase_cnt, on=['product_id'])

    if cp_type == 'click':
        g_purchase['rating'] = g_purchase['measure'] * g_purchase['cnt']
    else:
        g_purchase['rating'] = g_purchase['measure_purchase'] * g_purchase['cnt']

    g_pp = pd.merge(product_data, g_purchase, on=['product_id'], how='left', suffixes=('_prod', '_pur'))
    g_pp.fillna(0, inplace=True)
    g_pp = g_pp[g_pp['user_id'] != 0]
    g_pp['rating'] = g_pp['rating_prod'] + 1 + g_pp['rating_pur']

    user_prod_rating = g_pp.pivot_table('rating', index='user_id', columns='product_id')
    user_prod_rating.fillna(0, inplace=True)
    return user_prod_rating


def purchase_pop_rate(purchase_data):
    """
    전체 구매 데이터 중 특정 품목이 차지 하는 비율(얼마나 주목받는지)
    :param purchase_data: purchase_data
    :return: pur_rate 컬럼이 추가된 purchase_data
    """
    popular_desc = purchase_data.groupby(['product_id'], as_index=False).count()
    popular_desc.sort_values(by='measure', ascending=False, inplace=True)
    popular_desc.reset_index(drop=True, inplace=True)
    popular_desc['tot_pur'] = len(popular_desc)
    popular_desc = popular_desc[['product_id', 'user_id', 'tot_pur']]
    popular_desc.rename(columns={'product_id': 'product_id', 'user_id': 'purchase_cnt',
                                 'tot_pur': 'tot_pur'}, inplace=True)
    popular_desc['pur_rate'] = popular_desc.apply(lambda x: x['purchase_cnt'] / x['tot_pur'] * 10, axis=1)
    return popular_desc


def click_pop_rate(click_data):
    """
    전체 클릭 데이터 중 특정 품목이 차지 하는 비율(얼마나 주목받는지)
    :param click_data: click_data
    :return: click_rate 컬럼이 추가된 click_data
    """
    click_desc = click_data[['product_id', 'measure']].groupby(['product_id'], as_index=False).sum()
    click_desc.sort_values(by='measure', ascending=False, inplace=True)
    click_desc.reset_index(drop=True, inplace=True)

    click_desc['tot_click'] = len(click_desc)
    click_desc['click_rate'] = click_desc.apply(lambda x: x['measure'] / x['tot_click'] * 1000, axis=1)
    return click_desc


def weighted_popular_product(purchase_data, click_data):
    """
    pur_cnt, measure, click_rate, pur_rate를 이용 pop_rate 생성 후
    pop_rate에 min_max scale 후 log를 취함 -> standard scale이 더 성능이 좋아보임
    log를 취하는 이유는 좀 더 부드럽게 만들기 위함임

    :param purchase_data: purchase_data
    :param click_data: click_data
    :return: pd.DataFrame.columns = ['product_id', 'pop_rate']
    """
    scaler = StandardScaler()

    popular_desc, click_desc = purchase_pop_rate(purchase_data), click_pop_rate(click_data)

    wpp = pd.merge(click_desc, popular_desc, on='product_id', how='left').fillna(0)
    wpp['pop_rate'] = wpp.apply(lambda x: (x['purchase_cnt'] / x['measure']) * x['click_rate'] * x['pur_rate'], axis=1)
    wpp.sort_values('pop_rate', ascending=False, inplace=True)
    wpp[['pop_rate']] = scaler.fit_transform(wpp[['pop_rate']])
    wpp['pop_rate'] = wpp['pop_rate'].apply(lambda x: math.log(x+1))
    wpp = wpp[['product_id', 'pop_rate']]
    wpp = wpp.reset_index(drop=True)

    return wpp.groupby('product_id', as_index=False).sum()


def preprocessing_pur_click_data(purchase_data, click_data, product_data):
    """
    클릭 후 구매 데이터와 product에 대한 상대적 가격 점수를 구하기 위한 함수

    :param purchase_data: purchase_data
    :param click_data: click_dat
    :param product_data: product_data
    :return: purchase_after_click data, own_rating_product
    """
    weighted_popular = weighted_popular_product(purchase_data, click_data)

    own_rating_products = pd.merge(product_data, weighted_popular, on='product_id')
    own_rating_products.fillna(0, inplace=True)
    own_rating_products['rating'] = own_rating_products['relative_price'] + own_rating_products['pop_rate']

    # 실 구매에선 없는 카테고리도 있음. 해당 카테고리의 유저들에겐 단순 클릭 데이터로 추천을 해 줄 것임.
    p_after_c = purchase_after_click(click_data, purchase_data)
    p_after_c = duplicated_click(p_after_c)
    return p_after_c, own_rating_products


def new_user_preprocess(user_data, click_data, purchase_data):
    """
    new user recommendation을 위해 click_data와 purchase_data의 user_id를
    category화 하는 함수

    :param user_data: user_data
    :param click_data: click_data
    :param purchase_data: purchase_data
    :return: categorized click & purchase data
    """
    remain_col_list, rename_col_dict = ['user_id_y', 'product_id', 'measure', 'dt'], {'user_id_y': 'user_id'}

    cat_user_click = pd.merge(click_data, user_data, left_on='user_id', right_on='old')
    cat_user_pur = pd.merge(purchase_data, user_data, left_on='user_id', right_on='old')

    cat_user_click = cat_user_click[remain_col_list]
    cat_user_pur = cat_user_pur[remain_col_list]

    cat_user_click.rename(columns=rename_col_dict, inplace=True)
    cat_user_pur.rename(columns=rename_col_dict, inplace=True)
    return cat_user_click, cat_user_pur
