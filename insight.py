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

# 유저 카테고리별 가격에 대한 반응





# 유저 카테고리별 구매율(구매까지 이어지는지 클릭만 해보고 가는지


