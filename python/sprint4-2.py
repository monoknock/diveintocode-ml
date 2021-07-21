import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


# class MyLogisticRegression:
#     def __init__(self, n_classes, n_futures):
#         # ロジスティック回帰なら全ての要素を１に初期化しても良いが、NNの場合は各ユニットのパラメータが
#         # 同じ値に収束してしまうので、ランダムな値に設定する必要がある
#         self.W = np.ones((n_futures, n_classes)) #  [特徴量の数]×[分類クラス数]の行列
#         self.b = np.ones((1, n_classes)) # １×[分類クラス数]の行列


def init_parameter(n_classes, n_futures):
    w = np.ones((n_futures, n_classes))
    b = np.ones((1, n_classes))
    return w, b


def linear_trans(x, w, b):
    return np.dot(x, w) + b


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1)[:, np.newaxis]


def get_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def get_optimized_parameter(x, y_true, y_pred, w, b, alpha):
    w -= alpha * np.dot(x.T, y_pred - y_true) / x.shape[0]
    b -= alpha * np.dot(np.ones((1, x.shape[0])), y_pred - y_true) / x.shape[0]
    return w, b


def get_predict(y_pred_proba):
    y_arg_max = y_pred_proba.argmax(axis=1)
    y_pred = np.zeros(y_pred_proba.shape)
    for i in range(y_pred_proba.shape[0]):
        y_pred[i, y_arg_max[i]] = 1
    return y_pred


def get_accuracy(y, y_pred_proba):
    y_pred = get_predict(y_pred_proba)
    count = 0
    for i in range(len(y)):
        if (y[i] == y_pred[i]).all():
            count += 1
    return count / len(y)


def plot_iteration(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel("iterartion", fontsize=20)
    ax1.set_ylabel("loss", fontsize=20)
    ax1.plot(train_loss_list, label="train")
    ax1.plot(valid_loss_list, label="valid")
    ax1.legend(fontsize=20)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel("iterartion", fontsize=20)
    ax2.set_ylabel("accuracy", fontsize=20)
    ax2.plot(train_acc_list, label="train")
    ax2.plot(valid_acc_list, label="valid")
    ax2.legend(fontsize=20)
    # plt.savefig("./loss-acc-iteration.png")
    plt.show()


def optimize(x_train, y_train, x_valid, y_valid, alpha, iter_max):
    # 教師データのロス
    train_loss_list = []
    # 教師データの精度
    train_acc_list = []
    # テストデータのロス
    valid_loss_list = []
    # テストデータの精度
    valid_acc_list = []

    # W,bの初期化
    print(len(x_train.shape))
    print(len(y_train.shape))
    w, b = init_parameter(y_train.shape[1], x_train.shape[1])
    # 予測結果の初期値
    y_train_pred = softmax(linear_trans(x_train, w, b))

    for count in range(iter_max):
        # 教師データで最適化
        w, b = get_optimized_parameter(x_train, y_train, y_train_pred, w, b, alpha)

        # 教師データと検証データを最適化したW,bで予測
        y_train_pred = softmax(linear_trans(x_train, w, b))
        y_valid_pred = softmax(linear_trans(x_valid, w, b))

        # 教師データと検証データのロスを保存
        train_loss_list.append(get_loss(y_train, y_train_pred))
        valid_loss_list.append(get_loss(y_valid, y_valid_pred))

        # 教師データと検証データの精度を保存
        train_acc_list.append(get_accuracy(y_train, y_train_pred))
        valid_acc_list.append(get_accuracy(y_valid, y_valid_pred))

    plot_iteration(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list)

    return w, b


def get_split_data(x, y, size_ratio):
    # データ数
    n_samples = x.shape[0]
    # 分類クラス数
    n_classes = len(set(list(y)))

    # 教師データ最大index
    n_train_max = round(n_samples * size_ratio[0])
    # 検証データ最大index
    n_valid_max = n_train_max + round(n_samples * size_ratio[1])
    # テストデータ最大index
    n_test_max = n_valid_max + round(n_samples * size_ratio[2])

    # シャッフル
    np.random.seed(0)
    p = np.random.permutation(n_samples)
    x_shuffle = x[p, :]
    y_shuffle = y[p]

    # 教師データ
    x_train = x_shuffle[0:n_train_max, :]
    y_train = np.identity(n_classes)[y_shuffle[0:n_train_max]]

    # 検証データ
    x_valid = x_shuffle[n_train_max:n_valid_max, :]
    y_valid = np.identity(n_classes)[y_shuffle[n_train_max:n_valid_max]]

    # テストデータ
    x_test = x_shuffle[n_valid_max:n_test_max, :]
    y_test = np.identity(n_classes)[y_shuffle[n_valid_max:n_test_max]]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def main():
    # データ準備
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, )
    # fig, ax = plt.subplots(5, 10, figsize=(10, 6))
    # for i in range(0, 50):
    #     x = i % 10
    #     y = i // 10
    #     ax[y, x].set_xticks([])
    #     ax[y, x].set_yticks([])
    #     ax[y, x].set_title(mnist.target[i], fontsize=10)
    #     ax[y, x].imshow(mnist.data[i].reshape(28, 28), cmap='Greys')
    # plt.savefig("./mnist_evaluate-ec.png", bbox_inches='tight', pad_inches=0)
    # plt.show()
    # MNISTデータの一部を取得
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_split_data(
        mnist.data / 255, mnist.target.astype(int), (0.02, 0.005, 0.005))

    # 学習/推測
    alpha = 0.1
    iter_max = 1000
    w, b = optimize(x_train, y_train, x_valid, y_valid, alpha, iter_max)
    y_predict_proba = softmax(linear_trans(x_test, w, b))
    print(get_accuracy(y_test, y_predict_proba))


def main2():
    # データ準備
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    x = x.loc[:, ["sepal_length", "petal_length"]]
    y = pd.Series(iris.target, name="y")
    df_train = x.join(y).query('y in (1,2)')
    x = df_train.drop(["y"], axis=1)
    y = df_train["y"]
    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.25, shuffle=True, random_state=0)

    # 学習/推定
    alpha = 0.1
    iter_max = 1000
    w, b = optimize(x_train, y_train, x_test, y_test, alpha, iter_max)
    y_predict_proba = softmax(linear_trans(x_test, w, b))
    print(get_accuracy(y_test, y_predict_proba))


if __name__ == '__main__':
    main()
    # main2()
