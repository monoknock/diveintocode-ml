import numpy as np
import matplotlib.pyplot as plt

# データ点の個数
N = 100

# データ点のために乱数列を固定
np.random.seed(0)

# ランダムな N×2 行列を生成 = 2次元空間上のランダムな点 N 個
X = np.random.randn(N, 2)

def f(x, y):
    return 5 * x + 3 * y - 1  #  真の分離平面 5x + 3y = 1

T = np.array([ 1 if f(x, y) > 0 else 0 for x, y in X])

# 特徴関数
def phi(x, y):
    return np.array([x, y, 1])

# シグモイド関数
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

np.random.seed() # 乱数を初期化
w = np.random.randn(3)  # パラメータをランダムに初期化

# 学習率の初期値
eta = 0.1

for i in range(50):
    a = [j for j in range(N)]
    np.random.shuffle(a)

    for n in a:
        x_n, y_n = X[n, :]
        t_n = T[n]

        # 予測確率
        feature = phi(x_n, y_n)
        predict = sigmoid(np.inner(w, feature))
        w -= eta * (predict - t_n) * feature

    # イテレーションごとに学習率を小さくする
    eta *= 0.9

# 図を描くための準備
seq = np.arange(-3, 3, 0.01)
xlist, ylist = np.meshgrid(seq, seq)
zlist = [sigmoid(np.inner(w, phi(x, y))) for x, y in zip(xlist, ylist)]

# 散布図と予測分布を描画
plt.imshow(zlist, extent=[-3,3,-3,3], origin='lower', cmap=plt.cm.PiYG_r)
plt.plot(X[T==1,0], X[T==1,1], 'o', color='red')
plt.plot(X[T==0,0], X[T==0,1], 'o', color='blue')
plt.show()