import math
import pandas as pd



user_data = pd.read_csv('./data/_users.csv')
click_data = pd.read_csv('./data/_click_log.csv')
purchase_data = pd.read_csv('./data/_purchase_log.csv')
product_data = pd.read_csv('./data/_products.csv')
test_data = pd.read_csv('./data/_20230401_users.csv')

# 겹치는(users, 20230401_users) 유저 수 215
merged_data = pd.merge(user_data, test_data, on='user_id')
len(merged_data['user_id'].unique())

# test_set(_20230401_users) 중 구매 이력 있는 유저 수 11
t_purchase_data = pd.merge(purchase_data, test_data, on='user_id')
len(t_purchase_data['user_id'].unique())

# test_set(_20230401_users) 중 클릭 이력 있는 유저 수 215
t_click_data = pd.merge(click_data, test_data, on='user_id')
len(t_click_data['user_id'].unique())

# train_set(_users) 중 이력 있는 유저 수 618
m_purchase_data = pd.merge(purchase_data, user_data, on='user_id')
len(m_purchase_data['user_id'].unique())

# train_set(_users) 중 클릭 이력 있는 유저 수 49356
m_click_data = pd.merge(click_data, user_data, on='user_id')
len(m_click_data['user_id'].unique())

len(product_data['category'].unique()) # unique category 299개
g_product = product_data.groupby('category')['price'].describe()
# DIY,工具 接着,補修, DIY,工具 研磨,潤滑, DIY,工具 産業工具, DIY,工具 電動工具 등 비슷하게 묶을 수 있어 보이지만 현재 카테고리 내부에서도
# 편차가 크기 때문에 더 이상 카테고리를 줄이지 않고 진행.


len(product_data['product_id'].unique()) #48533
len(purchase_data['product_id'].unique()) # 432
len(click_data['product_id'].unique()) # 49303

for p in purchase_data['product_id'].unique():
    if p not in product_data['product_id'].unique():
        print(p) # 25782, 38776 product 메타에 없음

for p in click_data['product_id'].unique():
    if p not in product_data['product_id'].unique():
        print(p) # click에 product 에는 없는 애들이 많음

import ast
result = pd.read_csv('./data/result.csv')
result = result[['user_id', 'age_range', 'gender', 'recommendation']]
result['recommendation'] = result['recommendation'].apply(lambda x: ast.literal_eval(x))
result = result.explode('recommendation')


t_purchase_data = pd.merge(purchase_data, test_data, on='user_id')
t_purchase_data = pd.merge(t_purchase_data, product_data[['product_id', 'category']], on='product_id')

result = pd.merge(result, to_predict_user_data, left_on='user_id', right_on='old', how='left')
ex = result[result['user_id_x'].isin(t_purchase_data['user_id'])]
ex = pd.merge(ex, product_data[['product_id', 'category']], left_on='recommendation', right_on='product_id')


m_p = pd.merge(purchase_data, user_data, left_on='user_id', right_on='old')
mpp = pd.merge(m_p, product_data, on='product_id')
mpp = mpp[['user_id_y', 'product_id', 'category']]
mpp.drop_duplicates(subset=['user_id_y', 'product_id','category'], inplace=True)

nex = result[result['user_id_y'].isin(mpp['user_id_y'])]
nex = pd.merge(nex, product_data[['product_id', 'category']], left_on='recommendation', right_on='product_id')
nex = nex[['user_id_y', 'user_id_x', 'product_id', 'category']]
nex = nex[~nex['user_id_x'].isin(ex['user_id_x'].unique())]
nex.drop_duplicates(subset=['user_id_y', 'product_id','category'], inplace=True)

m_c = pd.merge(click_data, user_data, left_on='user_id', right_on='old')
mcp = pd.merge(m_c, product_data, on='product_id')
mcp = mcp[['user_id_y', 'product_id', 'category']]
mcp = mcp[mcp['user_id_y'].isin(not_in_cat)]
mcp.drop_duplicates(subset=['user_id_y', 'product_id','category'], inplace=True)

nex = result[result['user_id_y'].isin(mcp['user_id_y'])]
nex = pd.merge(nex, product_data[['product_id', 'category']], left_on='recommendation', right_on='product_id')
nex = nex[['user_id_y', 'user_id_x', 'product_id', 'category']]
nex = nex[~nex['user_id_x'].isin(ex['user_id_x'].unique())]
nex.drop_duplicates(subset=['user_id_y', 'product_id','category'], inplace=True)


aa = nex[['user_id_y', 'product_id']].groupby('user_id_y', as_index=False).count()




