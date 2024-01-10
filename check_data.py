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
    product_data['relative_price'] = product_data.apply(lambda x:
                                                        math.log(math.exp((x['price'] - x['mean'])/x['mean']) + 1),
                                                        axis=1)
    return product_data

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
    popular_desc = purchase_data.groupby(['product_id'], as_index=False).count()
    popular_desc.sort_values(by='measure', ascending=False, inplace=True)
    popular_desc.reset_index(drop=True, inplace=True)
    popular_desc['tot_pur'] = len(popular_desc)
    popular_desc = popular_desc[['product_id', 'user_id', 'tot_pur']]
    popular_desc.rename(columns={'product_id': 'product_id', 'user_id': 'purchase_cnt',
                                 'tot_pur': 'tot_pur'}, inplace=True)
    popular_desc['pur_rate'] = popular_desc.apply(lambda x: x['purchase_cnt']/x['tot_pur'] * 10, axis=1)

    # click to purchase. based on prd_id
    click_desc = click_data[['product_id', 'measure']].groupby(['product_id'], as_index=False).sum()
    click_desc.sort_values(by='measure', ascending=False, inplace=True)
    click_desc.reset_index(drop=True, inplace=True)
    click_desc['tot_click'] = len(click_desc)
    click_desc['click_rate'] = click_desc.apply(lambda x: x['measure']/x['tot_click'] * 1000, axis=1)

    wpp = pd.merge(click_desc, popular_desc, on='product_id', how='left').fillna(0)
    wpp['pop_rate'] = wpp.apply(lambda x: (x['purchase_cnt']/x['measure']) * x['click_rate'] * x['pur_rate'], axis=1)
    wpp.sort_values('pop_rate', ascending=False, inplace=True)
    return wpp.reset_index(drop=True, inplace=True)



purchase_data[purchase_data['user_id']==10888]

11069

