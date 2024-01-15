import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


user_data = pd.read_csv('./data/_users.csv')
click_data = pd.read_csv('./data/_click_log.csv')
purchase_data = pd.read_csv('./data/_purchase_log.csv')
product_data = pd.read_csv('./data/_products.csv')

age_2_idx = {'-19': 0, '109': 1, '20-29': 2, '30-39': 3, '40-49': 4, '50-59': 5, '609': 6}
gender_2_idx = {'3': 0, 'f': 1, 'm': 2}
product_2_idx = {cat: idx for idx, cat in enumerate(sorted(product_data['category'].unique()))}

# user data cat 화
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

cat_user = deepcopy(user_data)
cat_user = user_age_recat(cat_user)
cat_user['gender'].fillna('3', inplace=True)
cat_user['gender'].replace('unknown', '3', inplace=True)
cat_user['cat_age_range'] = cat_user['age_range'].map(age_2_idx)
cat_user['cat_gender'] = cat_user['gender'].map(gender_2_idx)
cat_user['cat'] = cat_user.apply(lambda x: str(x['cat_age_range']) + str(x['cat_gender']), axis=1)
cat_user.rename(columns={'user_id': 'old', 'cat': 'user_id'}, inplace=True)
cat_user = cat_user[['old', 'user_id']]
user_data = pd.merge(user_data, cat_user, left_on='user_id', right_on='old')

# product cat 화
cat_product = deepcopy(product_data)
cat_product['category'] = cat_product['category'].map(product_2_idx)


# 제품 카테고리별 클릭 구매 전환율
p_after_c = pd.merge(click_data, purchase_data, on=['user_id', 'product_id'], suffixes=('_click', '_purchase'))
p_after_c = p_after_c.where(p_after_c['dt_click'] <= p_after_c['dt_purchase'])
p_after_c = p_after_c.sort_values(by=['user_id', 'product_id', 'dt_click', 'dt_purchase'],
                                  ascending=True).reset_index(drop=True)
p_after_c.drop_duplicates(['user_id', 'product_id', 'dt_click'], inplace=True)

condition_1 = [['user_id', 'product_id', 'dt_purchase'], ['user_id', 'product_id', 'dt_purchase', 'measure_click']]
condition_2 = [['user_id', 'product_id', 'dt_click', 'dt_purchase'],
               ['user_id', 'product_id', 'dt_purchase', 'measure_purchase']]

p_after_c_1 = p_after_c.groupby(condition_1[0], as_index=False).sum()
p_after_c_2 = p_after_c.groupby(condition_2[0], as_index=False).sum()
p_after_c = pd.merge(p_after_c_1[condition_1[1]], p_after_c_2[condition_2[1]],
                     on=['user_id', 'product_id', 'dt_purchase'])
p_after_c.drop_duplicates(['user_id', 'product_id', 'dt_purchase', 'measure_click', 'measure_purchase'],
                          inplace=True)
p_after_c = pd.merge(p_after_c, cat_product, on='product_id')
p_after_c = p_after_c[['user_id', 'category', 'dt_purchase', 'measure_click','measure_purchase']]
p_after_c['cat'] = p_after_c['category'].map({val: key for key, val in product_2_idx.items()})

# 카테고리별 구매전환율
p_after_c_rate = p_after_c.groupby(['category'], as_index=False).sum()
p_after_c_rate['purchase_rate'] = p_after_c_rate['measure_purchase'] / p_after_c_rate['measure_click']
p_after_c_rate = p_after_c_rate[['category', 'measure_click', 'measure_purchase', 'purchase_rate']]
p_after_c_rate['cat'] = p_after_c_rate['category'].map({val: key for key, val in product_2_idx.items()})
aa = p_after_c_rate.sort_values(by='measure_purchase', ascending=False).iloc[:10,:]
np.mean(p_after_c_rate['purchase_rate'])
plt.bar(aa['category'], aa['measure_purchase'])
plt.title('Purchase top 10', fontsize=20)
plt.xlabel('Product Category')
plt.ylabel('measure')
plt.show()

p_after_c_rate = p_after_c.groupby(['category'], as_index=False).sum()
p_after_c_rate['purchase_rate'] = p_after_c_rate['measure_purchase'] / p_after_c_rate['measure_click']
p_after_c_rate = p_after_c_rate[['category', 'measure_click', 'measure_purchase', 'purchase_rate']]
p_after_c_rate = p_after_c_rate[p_after_c_rate['measure_click'] > 2]
p_after_c_rate['cat'] = p_after_c_rate['category'].map({val: key for key, val in product_2_idx.items()})
index = np.arange(len(p_after_c_rate['cat'].unique()))
plt.bar(index, p_after_c_rate['purchase_rate'])
plt.title('Purchase after Click Rate per Category', fontsize=20)
plt.xlabel('Product Category')
plt.ylabel('Purchase after Click Rate')
plt.show()

p_after_c = pd.merge(p_after_c, cat_user[['old', 'user_id']], left_on='user_id', right_on='old')
p_after_c = p_after_c[['user_id_y', 'category', 'dt_purchase', 'measure_click', 'measure_purchase', 'old']]
p_after_c.columns = ['cat_user_id', 'category', 'dt_purchase', 'measure_click', 'measure_purchase', 'user_id']
p_after_c = p_after_c.groupby(['cat_user_id', 'category'], as_index=False).sum()
p_after_c['puchase_rate'] = p_after_c['measure_purchase'] / p_after_c['measure_click']
p_after_c = p_after_c[['cat_user_id', 'category', 'puchase_rate']]

# 유저 카테고리 별 관심 카테고리

cat_click = pd.merge(click_data, cat_product, on='product_id')
cat_user_c = pd.merge(cat_click, cat_user, left_on='user_id', right_on='old')
cat_user_c = cat_user_c[['user_id_x', 'product_id', 'measure', 'price', 'category', 'user_id_y']]
cat_user_c.columns = ['user_id', 'product_id', 'measure', 'price', 'category', 'cat_user']
g_cat_user = cat_user[['user_id', 'user_id_x']].groupby('user_id', as_index=False).count()
g_cat_user = pd.merge(g_cat_user, cat_user[['age_range', 'gender', 'user_id']], left_on='user_id', right_on='user_id')
g_cat_user.drop_duplicates(subset=g_cat_user.columns, inplace=True)
g_cat_user['age_range'].replace('609', '60-', inplace=True)
g_cat_user['age_range'].replace('109', 'unknown', inplace=True)
g_cat_user['gender'].replace('3', 'unknown', inplace=True)
g_cat_user.columns = ['user_id', 'cnt', 'age_range', 'gender']


# 유저 카테고리별 활동량
g_cat_user_c = cat_user_c[['cat_user', 'measure']].groupby('cat_user', as_index=False).sum()
g_cat_user_c = pd.merge(g_cat_user_c, cat_user[['age_range', 'gender', 'user_id']], left_on='cat_user', right_on='user_id')
g_cat_user_c.drop_duplicates(subset=g_cat_user_c.columns, inplace=True)
g_cat_user_c['age_range'].replace('609', '60-', inplace=True)
g_cat_user_c['age_range'].replace('109', 'unknown', inplace=True)
g_cat_user_c['gender'].replace('3', 'unknown', inplace=True)
g_cat = pd.merge(g_cat_user_c, g_cat_user, left_on='cat_user', right_on='user_id')
g_cat['rate'] = g_cat['measure'] / g_cat['cnt']


# 유저 카테고리별 관심 카테고리(click)
cc_measure = cat_user_c[['cat_user', 'product_id', 'category', 'measure']].groupby(['cat_user', 'product_id', 'category'], as_index=False).sum()
result_list = []
for cat_userid in (cc_measure['cat_user'].unique()):
    result = cc_measure[(cc_measure['cat_user']==cat_userid) &
                        (cc_measure['measure']==max(cc_measure[cc_measure['cat_user']==cat_userid]['measure']))]
    result_list.append(result)

# max(cc_measure[cc_measure['cat_user']=='02']['measure'])
result = pd.concat(result_list)
result['cat'] = result['category'].map({val: key for key, val in product_2_idx.items()})
cat_user = deepcopy(user_data)
cat_user = user_age_recat(cat_user)
cat_user['gender'].fillna('3', inplace=True)
cat_user['gender'].replace('unknown', '3', inplace=True)
cat_user['cat_age_range'] = cat_user['age_range'].map(age_2_idx)
cat_user['cat_gender'] = cat_user['gender'].map(gender_2_idx)
cat_user['cat'] = cat_user.apply(lambda x: str(x['cat_age_range']) + str(x['cat_gender']), axis=1)
cat_user.rename(columns={'user_id': 'old', 'cat': 'user_id'}, inplace=True)
result = pd.merge(result, cat_user[['age_range', 'gender', 'user_id']], left_on='cat_user', right_on='user_id')
result.drop_duplicates(subset=result.columns, inplace=True)
result['age_range'].replace('609', '60-', inplace=True)
result['age_range'].replace('109', 'unknown', inplace=True)
result['gender'].replace('3', 'unknown', inplace=True)


#############################
# 유저 카테고리별 가격에 대한 반응
#############################
cat_pur = pd.merge(purchase_data, cat_product, on='product_id')
cat_user.columns = ['user_id', 'age_range', 'gender', 'cat_age_range', 'cat_gender', 'cat_user']
cat_pur = pd.merge(cat_pur, cat_user, on='user_id')
cat_pur = cat_pur[['user_id', 'product_id', 'measure', 'price', 'category', 'age_range', 'gender', 'cat_user']]
cat_pur['age_range'].replace('609', '60-', inplace=True)
cat_pur['age_range'].replace('109', 'unknown', inplace=True)
cat_pur['gender'].replace('3', 'unknown', inplace=True)

# 유저 카테고리별 구매
# 203040 남자들이 스마트폰 종류를 지나치게 많이 구매 정확한 선호를 반영한다고 할 수 없음.
# 가격이 10만이 넘어가는 것들은 10개 안쪽으로 구매해야 정상적 선호라고 보고 데이터 반영할 것
cat_pur_user_g = cat_pur[['user_id', 'cat_user', 'product_id', 'category', 'measure']].groupby(['user_id', 'cat_user', 'product_id', 'category'], as_index=False).sum()
cat_pur_user_g = pd.merge(cat_pur_user_g, cat_pur[['user_id', 'product_id', 'category', 'price', 'age_range', 'gender']], on=['user_id', 'product_id', 'category'])
per_user_g = (cat_pur_user_g[['cat_user', 'product_id', 'category', 'measure', 'price','age_range', 'gender']].
              groupby(['cat_user', 'product_id', 'category', 'price','age_range', 'gender'], as_index=False).sum())
per_user_g['cat'] = per_user_g['category'].map({val: key for key, val in product_2_idx.items()})
mask1 = (per_user_g.price >= 100000) & (per_user_g.measure > 10)
per_user_g = per_user_g.loc[~mask1,:]
# 거의 모든 연령대의 성별이 스마트폰을 가장 많이 구매함.
# 한가지 특이할만한 점은 아기용품에 대한 클릭 집중도가 높았던 30대 여성 집단에선 아기 용품을 별로 구매하지 않았단 것임
# 아기 관련 용품이 다른 구매채널보다 매력적인지 아닌지 확인해 볼 필요가 있음(종류, 가격 등)
gg_user = per_user_g[['cat_user', 'category', 'age_range', 'gender', 'measure', 'cat']].groupby(['cat_user', 'category', 'age_range', 'gender', 'cat'], as_index=False).sum()
aa = gg_user[gg_user['cat_user']=='31']

# 유저 카테고리별 가격 민감도
product_mean = cat_product[['category', 'price']].groupby('category', as_index=False).mean()
product_mean.columns = ['category', 'price_mean']
p_mean_per_user = pd.merge(per_user_g, product_mean, on='category')
p_mean_per_user['price_deviation'] = (p_mean_per_user['price'] - p_mean_per_user['price_mean']) * p_mean_per_user['measure']
mps_user = deepcopy(p_mean_per_user)
# mps_user = p_mean_per_user[p_mean_per_user['category']==119]
mps_user = mps_user[['cat_user', 'category', 'age_range', 'gender', 'measure', 'cat', 'price_mean', 'price_deviation']]
mps_user = mps_user.groupby(['cat_user', 'category', 'age_range', 'gender', 'cat', 'price_mean'], as_index=False).sum()
# 대체적으로 남성들이 스마트폰 product category 내에서 비싼 제품을 사고 여성들이 상대적으로 덜 비싼 제품을 구매함 - 스마트폰
# 여성의복은 전 세대의 여성들에게 약간의 구매가 있지만 남성패션은 남성들의 구매가 단 한건도 존재하지 않음 - 옷
# category 내 product 다양성이 고려되진 않았음
mps_user_sum = mps_user[['category', 'cat', 'price_deviation']].groupby(['category', 'cat'], as_index=False).sum()
mps_user_sum.columns = ['category', 'cat', 'sum_deviation']
mps_user_cnt = mps_user[['category', 'cat', 'cat_user']].groupby(['category', 'cat'], as_index=False).count()
mps_user_cnt.columns = ['category', 'cat', 'cnt']
mps_user = pd.merge(mps_user, mps_user_sum, on=['category', 'cat'])
mps_user = pd.merge(mps_user, mps_user_cnt, on=['category', 'cat'])
mps_per_user = mps_user[['cat_user', 'category', 'age_range', 'gender','cat', 'measure', 'price_deviation']]
mps_per_user_minus = mps_per_user[mps_per_user['price_deviation'] < 0].groupby(['cat_user', 'age_range', 'gender'], as_index=False).count()
mps_per_user_plus = mps_per_user[mps_per_user['price_deviation'] >= 0].groupby(['cat_user', 'age_range', 'gender'], as_index=False).count()
mps_per_user_minus = mps_per_user_minus[['cat_user', 'age_range', 'gender', 'category']]
mps_per_user_plus = mps_per_user_plus[['cat_user', 'age_range', 'gender', 'category']]
mps_per_user_minus.columns = ['cat_user', 'age_range', 'gender', 'minus_cnt']
mps_per_user_plus.columns = ['cat_user', 'age_range', 'gender', 'plus_cnt']
mps_pmp = pd.merge(mps_per_user_minus, mps_per_user_plus, on=['cat_user', 'age_range', 'gender'], how='left')
# 대체적으로 203040 남성들이 product카테고리 평균보다 비싼 product를 사며 여성들은 평균보다 싼 product를 사는 경향이 있음
# 하지만 이는 실 구매가 1건만 있는 경우도 포함이 된 결과기 때문에 지속적인 관찰이 필요해 보임

#####################
# 유저 카테고리별 구매율(구매까지 이어지는지 클릭만 해보고 가는지)
#####################
g_cat_user_c = cat_user_c[['cat_user', 'measure']].groupby('cat_user', as_index=False).sum()
g_cat_user_c = pd.merge(g_cat_user_c, cat_user[['user_id', 'gender', 'age_range']], left_on='cat_user', right_on='user_id')
g_cat_user_c = g_cat_user_c[['cat_user', 'measure', 'gender', 'age_range']]
g_cat_user_c['age_range'].replace('609', '60-', inplace=True)
g_cat_user_c['age_range'].replace('109', 'unknown', inplace=True)
g_cat_user_c['gender'].replace('3', 'unknown', inplace=True)
g_cat_user_c.drop_duplicates(inplace=True)

cat_user = cat_user[['old', 'age_range', 'gender', 'user_id']]
cat_user.columns = ['user_id', 'age_range', 'gender', 'cat_user']
p_after_c['measure_purchase'] = p_after_c['measure_purchase'] / p_after_c['measure_purchase']
cat_pur = pd.merge(p_after_c, cat_user, on='user_id')
cat_user_pur = cat_pur[['cat_user', 'measure_purchase', 'gender', 'age_range']].groupby(['cat_user', 'gender', 'age_range'], as_index=False).sum()
user_pur_click_rate = pd.merge(g_cat_user_c, cat_user_pur, on=['cat_user','gender', 'age_range'], how='left')
user_pur_click_rate.fillna(0, inplace=True)
user_pur_click_rate['click_pur_rate'] = user_pur_click_rate['measure_purchase'] / user_pur_click_rate['measure']
# 203040 남성들이 클릭:구매 전환비를 가지고 있음. 약 100번 클릭 시 한번 이상 사는

cat_pur.columns
cat_pur_user = cat_pur[['category', 'measure_click', 'measure_purchase', 'age_range', 'gender', 'cat_user']]
cat_pur_user = cat_pur_user.groupby(['category', 'age_range', 'gender', 'cat_user'], as_index=False).sum()
cat_pur_user['click_pur_rate'] = cat_pur_user['measure_purchase'] / cat_pur_user['measure_click']
cpp_pro = cat_pur_user[['category', 'cat_user']]
cpp_pro = cpp_pro.groupby('category', as_index=False).count()
cpp_pro = cpp_pro[cpp_pro['cat_user'] > 2]
cat_pur_user = pd.merge(cat_pur_user, cpp_pro, on='category')
cat_pur_user['cat'] = cat_pur_user['category'].map({val: key for key, val in product_2_idx.items()})
cat_pur_user.columns
cat_pur_user = cat_pur_user[['cat_user_x', 'age_range', 'gender', 'cat', 'measure_click', 'measure_purchase', 'click_pur_rate']]
# 구매가 최소 3개 이상의 유저 카테고리로 이루어진 물품카테고리 구매전환율(1명의 user만 구매한 product를 제외하기 위함)
# 여기 존재하지 않는 카테고리는 아예 클릭이 없었거나, 클릭했지만 구매가 없는경우

bb = cat_pur_user[cat_pur_user['cat_user_x']=='41']
aa = cat_pur_user[cat_pur_user['cat_user_x']=='31']
cc = cat_pur_user[cat_pur_user['cat_user_x']=='51']
dd = cat_pur_user[cat_pur_user['cat_user_x']=='61']
ee = cat_pur_user[cat_pur_user['cat_user_x']=='21']
# 특이하게 30대 여성보다 40대 여성에서 아기용품에 대해 구매율이 더 높음

cat_click_not = cat_click[~cat_click['category'].isin(cat_pur_user['category'].unique())]
cat_click_not = cat_click_not[['category', 'measure']]
cat_click_not = cat_click_not.groupby('category', as_index=False).sum()
cat_click_not['cat'] = cat_click_not['category'].map({val: key for key, val in product_2_idx.items()})
# pc 노트북, 게임, 애완동물 용품, 아이돌 사진집, 취미 수예 핸드크래프트 카테고리 등등이 많은 클릭에도 불구하고 구매로 이어지지 않음.
# 가격 하락이나 종류의 다양성 증가 등 구매를 올릴 수 있다면 매출이 증가할 가능성 있음
# 예를 들어 게임은 19세 이하 남성이 상품 카테고리중 가장 관심 있어 하는데 해당 카테고리에 대한 경쟁력을 강화하면 19세 이하 남성 고객을 확보 할 수 있음

cat_click_yes = cat_click[cat_click['category'].isin(cat_pur_user['category'].unique())]
cat_click_yes = cat_click_yes[['category', 'measure']]
cat_click_yes = cat_click_yes.groupby('category', as_index=False).sum()
cat_click_yes['cat'] = cat_click_yes['category'].map({val: key for key, val in product_2_idx.items()})
