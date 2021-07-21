import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


def standardize(x):
    return (x - mu) / sigma


# ベクトル同時の乗算を成立させるため、標本データに０列目を追加する
def add_x0(x):
    # 標本データの行数分、要素１の行データを生成する
    x0 = np.ones([x.shape[0],1])
    # 生成した行データを、標本データの直前に置いて結合する
    return np.hstack([x0,x])


# 識別関数としてのシグモイド関数を実装
def f(x):
    # θ群と変数群をそれぞれベクトルと見なし乗算を行なっている
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# 指定回数パラメータの更新を行う更新式を実装
def update_theta(train_y):
    new_theta = theta + ETA * np.dot(train_y - f(X), X)
    return new_theta


# 決定境界を可視化する関数を実装
def validator(theta):
    # (x,y)におけるy軸。
    # シグモイド関数では θTxa=0 が分類AとBの境界だったので、その時の値を求めてプロットする。
    # θTx = θ0x0 + θ1x1 + θ2x2 = 0
    # x2  = -(θ0 + θ1x1) / θ2
    y = -(theta[0] + theta[1] * xline) / theta[2]

    return y


# # 別途用意したcsvからデータを読み込む
# test = np.loadtxt('0226.csv', delimiter=',',skiprows=1)
# # 標本データの抽出。元データの２列を読み込み、以後ベクトルとして扱う
# train_x = test[:,0:2]
# # 正解値の抽出
# train_y = test[:,2]

# データ準備
iris = load_iris()
x = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
x = x.loc[:, ["sepal_length", "petal_length"]]
y = pd.Series(iris.target, name="y")
df_train = x.join(y).query('y in (1,2)')
x = df_train.drop(["y"], axis=1)
y = df_train["y"]
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.25, shuffle=True, random_state=0)




# パラメータの初期化。ベクトル同士の乗算を成立させるため、１行３列のデータを設定。
theta = np.random.rand(3)
# 標準化関数の実装
mu = x_train.mean() # 平均
sigma = x_train.std() # 分散
# 標本データの標準化。標準化を行うことで、データの平均を０に、分散を１に慣らすことができる
train_x = standardize(x_train)
# 標準化され、かつ０列目に要素１のデータが設定された標本データ
X = add_x0(train_x)
# 学習率の定義。厳密な規定はない。
ETA = 1e-3
# 更新回数の指定
epoch = 5000

print('更新前のパラメータθ')
print(theta)

# パラメータ更新の実行
for _ in range(epoch):
    theta = update_theta(y_train)

print('更新後のパラメータθ')
print(theta)

# 標本データの可視化
# 正解値が１のデータを'o'で可視化
plt.plot(train_x[y_train==1, 0],train_x[y_train==1,1],'o')
# 正解値が０のデータを'o'で可視化
plt.plot(train_x[y_train==0, 0],train_x[y_train==0,1],'x')

# 描画するグラフのx軸の長さ
xline = np.linspace(-2,2,100)

# 決定境界の可視化
# plt.plot(x0, -(theta[0] + theta[1] * xline) / theta[2], linestyle='solid')
plt.plot(xline, validator(theta), linestyle='solid', label='boudary')
plt.legend(loc='lower right')
plt.show()

print('算出された決定境界の式')
print('y = {:0.3f} + {:0.3f} * x1 + {:0.3f} * x2'.format(theta[0], theta[1], theta[2]))

#  分類確率の可視化
plt.plot(np.reshape(np.arange(X.shape[0]),16,1), np.sort(f(X))[::-1])
plt.ylabel('probability')
plt.xlabel('number')
plt.show()
