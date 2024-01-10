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










