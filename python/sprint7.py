import numpy as np
import itertools
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters, max_iter=1000, random_seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)

    def fit(self, x):
        # 指定したクラスター数分のラベルを繰り返し作成するジェネレータを生成（0,1,2,0,1,2,0,1,2...みたいな感じ）
        cycle = itertools.cycle(range(self.n_clusters))
        print(cycle)
        # 各データポイントに対してクラスタのラベルをランダムに割り振る
        self.labels_ = np.fromiter(itertools.islice(cycle, x.shape[0]), dtype=np.int)
        self.random_state.shuffle(self.labels_)
        labels_prev = np.zeros(x.shape[0])
        count = 0
        self.cluster_centers_ = np.zeros((self.n_clusters, x.shape[1]))

        # 各データポイントが属しているクラスターが変化しなくなった、又は一定回数の繰り返しを越した場合は終了
        while (not (self.labels_ == labels_prev).all() and count < self.max_iter):
            #その時点での各クラスターの重心を計算する
            for i in range(self.n_clusters):
                xx = x[self.labels_ == i, :]
                self.cluster_centers_[i, :] = xx.mean(axis = 0)
            # 各データポイントと各クラスターの重心間の距離を総当たりで計算する
            dist = ((x[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2).sum(axis = 1)
            # 1つ前のクラスターラベルを覚えておく。1つ前のラベルとラベルが変化しなければプログラムは終了する。
            labels_prev = self.labels_
            # 再計算した結果、最も距離の近いクラスターのラベルを割り振る
            self.labels_ = dist.argmin(axis = 1)
            count += 1

    def predict(self, X):
        dist = ((X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2).sum(axis = 1)
        labels = dist.argmin(axis = 1)
        return labels


# 適当なデータセットを作成する
np.random.seed(0)
points1 = np.random.randn(80, 2)
points2 = np.random.randn(80, 2) + np.array([4,0])
points3 = np.random.randn(80, 2) + np.array([5,8])

points = np.r_[points1, points2, points3]
np.random.shuffle(points)

# 3つのクラスタに分けるモデルを作成
model = KMeans(3)
model.fit(points)

print(model.labels_)