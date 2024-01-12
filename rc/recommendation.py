import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# base_src = 'drive/MyDrive/RecoSys/Data'
# u_data_src = os.path.join(base_src, 'u.data')
# r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
# # timestamp 제거
#
#
# # train set과 test set 분리
#
# TRAIN_SIZE = 0.8
# # (사용자-영화-평점)
# user_prod_rating.reset_index(inplace=True)
# ratings = shuffle(user_prod_rating, random_state=2021)
# cutoff = int(TRAIN_SIZE * len(ratings))
# ratings_train = ratings.iloc[:cutoff]
# ratings_test = ratings.iloc[cutoff:]


class MatrixFactorization:
    def __init__(self, ratings, hyper_params):
        self.R = np.array(ratings)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = hyper_params['K']
        self.alpha = hyper_params['alpha']
        self.beta = hyper_params['beta']
        self.iterations = hyper_params['iterations']
        self.verbose = hyper_params['verbose']
        item_id_index = []
        index_item_id = []
        for i, one_id, in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)

        user_id_index = []
        index_user_id = []
        for i, one_id, in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)

    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors ** 2))

    def sgd(self):
        for i, j, r in self.samples:
            # 사용자 i 아이템 j에 대한 평점 예측치 계산
            prediction = self.get_prediction(i, j)
            # 실제 평점과 비교한 오차 계산
            e = (r - prediction)

            # 사용자 평가 경향 계산 및 업데이트
            self.b_u[i] += self.alpha * (e - (self.beta * self.b_u[i]))
            # 아이템 평가 경향 계산 및 업데이트
            self.b_d[j] += self.alpha * (e - (self.beta * self.b_d[j]))

            # P 행렬 계산 및 업데이트
            self.P[i, :] += self.alpha * ((e * self.Q[j, :]) - (self.beta * self.P[i, :]))
            # Q 행렬 계산 및 업데이트
            self.Q[j, :] += self.alpha * ((e * self.P[i, :]) - (self.beta * self.Q[j, :]))

    def get_prediction(self, i, j):  # 평점 예측값 구하는 함수
        # 전체 평점 + 사용자 평가 경향 + 아이템에 대한 평가 경향 + i번쨰 사용자의 요인과 j번째 아이템 요인의 행렬 곱
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Test Set 선정
    def set_test(self, ratings_test):
        test_set = []
        for i in range(len(ratings_test)):
            x = self.user_id_index[ratings_test.iloc[i, 0]]  # 사용자 id
            y = self.item_id_index[ratings_test.iloc[i, 1]]  # 영화 id
            z = ratings_test.iloc[i, 2]  # 실제 평점
            test_set.append([x, y, z])
            self.R[x, y] = 0  # 안해주면 전체 데이터에 대한 트레이닝이 이루어짐
        self.test_set = test_set
        return test_set

    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])  # 사용자 ID, 영화 ID
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error / len(self.test_set))

    def test(self):
        self.P = np.random.normal(scale=1. / self.K,
                                  size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K,
                                  size=(self.num_items, self.K))
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])  # 테스트셋에서 0으로 처리하여서 트레이닝 셋만

        # 트레이닝 셋에 대해서 데이터셋 구성
        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            train_rmse = self.rmse()
            test_rmse = self.test_rmse()
            training_process.append((i + 1, train_rmse, test_rmse))
            if self.verbose:
                if (i + 1) % 10 == 0:
                    print('Iteration : %d ; Train RMSE = %.4f ; Test RMSE = %.4f' % (i + 1, train_rmse, test_rmse))
        return training_process

    def get_one_prediction(self, user_id, item_id):
        return self.get_prediction(self.user_id_index[user_id],
                                   self.item_id_index[item_id])

    def full_prediction(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_d[np.newaxis, :] + self.P.dot(self.Q.T)

    def recommend_prod_by_user(self, rating_matrix, user_id, top=5):
        pred_matrix = np.dot(self.P, self.Q.T)
        pred_df = pd.DataFrame(data=pred_matrix, index=rating_matrix.index, columns=rating_matrix.columns)

        user_rating = rating_matrix.loc[user_id, :]
        bought = user_rating[user_rating > 0].index.tolist()
        prod_list = rating_matrix.columns.tolist()
        unseen_list = [prod for prod in prod_list if prod not in bought]

        recomm_prod = pred_df.loc[user_id, unseen_list].sort_values(ascending=False)[:top]
        recomm_prod_df = pd.DataFrame(data=recomm_prod.values,
                                      index=recomm_prod.index,
                                      columns=['pred_score'])

        return recomm_prod_df



#
#
#
#
# R_temp = ratings.pivot(index='user_id',
#                        columns='movie_id',
#                        values='rating').fillna(0)
# hyper_params = {
#     'train_size': 0.8,
#     'K': 50,
#     'alpha': 0.001,
#     'beta': 0.002,
#     'iterations': 100,
#     'verbose': True
# }
# mf = MatrixFactorization(ratings, hyper_params)
# test_set = mf.set_test(ratings_test)
# result = mf.test()
# mf.recommend_prod_by_user(user_prod_rating, user_id='22')
#
#
#
# user_prod_rating.index


