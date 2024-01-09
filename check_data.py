import math
import pandas as pd



user_data = pd.read_csv('./data/_users.csv')
click_data = pd.read_csv('./data/_click_log.csv')
purchase_data = pd.read_csv('./data/_purchase_log.csv')
product_data = pd.read_csv('./data/_products.csv')
test_data = pd.read_csv('./data/_20230401_users.csv')

merged_data = pd.merge(user_data, test_data, on='user_id')
len(merged_data['user_id'].unique()) # 겹치는 유저 수 215

t_purchase_data = pd.merge(purchase_data, test_data, on='user_id')
len(t_purchase_data['user_id'].unique()) # 겹치는 구매이력 있는 유저 수 11

t_click_data = pd.merge(click_data, test_data, on='user_id')
len(t_click_data['user_id'].unique()) # 테스트 셋 중 클릭 이력 있는 유저 수 215

m_purchase_data = pd.merge(purchase_data, user_data, on='user_id')
len(m_purchase_data['user_id'].unique()) # 겹치는 구매이력 있는 유저 수 618

m_click_data = pd.merge(click_data, user_data, on='user_id')
len(m_click_data['user_id'].unique()) # train 셋 중 클릭 이력 있는 유저 수 49356

len(product_data['category'].unique()) # unique category 299개
g_product = product_data.groupby('category')['price'].describe()
# DIY,工具 接着,補修, DIY,工具 研磨,潤滑, DIY,工具 産業工具, DIY,工具 電動工具 등 비슷하게 묶을 수 있어 보이지만 현재 카테고리 내부에서도
# 편차가 크기 때문에 더 이상 카테고리를 줄이지 않고 진행.


# 카테고리 인덱스 화
def prd_category_to_idx(category_list):
    return {category: idx for idx, category in enumerate(set(category_list))}

product_data['cat_idx'] = product_data['category'].map(prd_category_to_idx(product_data['category']))


# 각 prd id의 가격이 cat 내에서 평균에서 얼마만큼 떨어져 있는지 -> 가격을 대체할 것임
def standard_price(product_data):
    g_products = product_data.groupby('cat_idx')['price'].describe().reset_index()
    product_data = pd.merge(product_data, g_products[['cat_idx', 'mean']], on='cat_idx')
    product_data['relative_price'] = product_data.apply(lambda x: math.log(math.exp(x['price'] - x['mean']) + 1), axis=1)





math.exp(1000 - 2000)

math.log(1)










