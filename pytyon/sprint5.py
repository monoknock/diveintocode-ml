from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class HardMarginSVM:
    """
    Attributes
    -------------
    eta : float
        学習率
    epoch : int
        エポック数
    random_state : int
        乱数シード
    is_trained : bool
        学習完了フラグ
    num_samples : int
        学習データのサンプル数
    num_features : int
        特徴量の数
    w : NDArray[float]
        パラメータベクトル: (num_features, )のndarray
    b : float
        切片パラメータ
    alpha : NDArray[float]
        未定乗数: (num_samples, )のndarray

    Methods
    -----------
    fit -> None
        学習データについてパラメータベクトルを適合させる
    predict -> NDArray[int]
        予測値を返却する
    """

    def __init__(self, eta=0.001, epoch=1000, random_state=42):
        self.eta = eta
        self.epoch = epoch
        self.random_state = random_state
        self.is_trained = False

    def fit(self, X, y):
        """
        学習データについてパラメータベクトルを適合させる

        Parameters
        --------------
        X : NDArray[NDArray[float]]
            学習データ: (num_samples, num_features)の行列
        y : NDArray[float]
            学習データの教師ラベル: (num_samples)のndarray
        """
        self.num_samples = X.shape[0]
        self.num_features = X.shape[1]
        # パラメータベクトルを0で初期化
        self.w = np.zeros(self.num_features)
        self.b = 0
        # 乱数生成器
        rgen = np.random.RandomState(self.random_state)
        # 正規乱数を用いてalpha(未定乗数)を初期化
        self.alpha = rgen.normal(loc=0.0, scale=0.01, size=self.num_samples)

        # 最急降下法を用いて双対問題を解く
        for _ in range(self.epoch):
            self._cycle(X, y)

        # サポートベクトルのindexを取得
        indexes_sv = [i for i in range(self.num_samples) if self.alpha[i] != 0]
        print(indexes_sv)
        # w を計算 (式1)
        for i in indexes_sv:
            self.w += self.alpha[i] * y[i] * X[i]
        # b を計算 (式2)
        for i in indexes_sv:
            self.b += y[i] - (self.w @ X[i])
        self.b /= len(indexes_sv)
        # 学習完了のフラグを立てる
        self.is_trained = True

    def predict(self, X):
        """
        予測値を返却する

        Parameters
        --------------
        X : NDArray[NDArray[float]]
            分類したいデータ: (any, num_features)の行列

        Returns
        ----------
        result : NDArray[int]
            分類結果 -1 or 1: (any, )のndarray
        """
        if not self.is_trained:
            raise Exception('This model is not trained.')

        hyperplane = X @ self.w + self.b
        result = np.where(hyperplane > 0, 1, -1)
        return result

    def _cycle(self, X, y):
        """
        勾配降下法の1サイクル

        Parameters
        --------------
        X : NDArray[NDArray[float]]
            学習データ: (num_samples, num_features)の行列
        y : NDArray[float]
            学習データの教師ラベル: (num_samples)のndarray
        """
        y = y.reshape([-1, 1])  # (num_samples, 1)の行列にreshape
        H = (y @ y.T) * (X @ X.T)  # (式3)
        # 勾配ベクトルを計算
        grad = np.ones(self.num_samples) - H @ self.alpha  # (式4)
        # alpha(未定乗数)の更新
        self.alpha += self.eta * grad
        # alpha(未定乗数)の各成分はゼロ以上である必要があるので負の成分をゼロにする
        self.alpha = np.where(self.alpha < 0, 0, self.alpha)


def main1(x_train, x_test, y_train, y_test):
    # svmのパラメータを学習
    hard_margin_svm = HardMarginSVM()
    hard_margin_svm.fit(x_train, y_train)

    pre = hard_margin_svm.predict(x_test)
    print(pre)


if __name__ == '__main__':
    # データの前準備
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    # x = x.loc[:, ["sepal_length", "petal_length"]]
    y = pd.Series(iris.target, name="y")
    df_train = x.join(y).query('y in (1,2)')
    x = df_train.drop(["y"], axis=1)
    y = df_train["y"]

    # xを標準化しないとエラーがでる
    # _cycle で成分がすべて負の数になっていることが原因
    sc = StandardScaler()
    x_std = sc.fit_transform(x)

    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_std, y.values, test_size=0.2, random_state=42, stratify=y)

    main1(x_train2, x_test2, y_train2, y_test2)


