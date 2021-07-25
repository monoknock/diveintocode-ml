import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

colors = ['#de3838', '#007bc3', '#ffd12a']

# area = [17.00, 18.00, 21.00, 24.00, 18.90, 20.66, 23.51, 25.21, 24.94, 30.22]  # 説明変数1
# age = [31, 34, 36, 31, 28, 22, 19, 12, 4, 0]  # 説明変数2
# rent = [35000, 36000, 39000, 42000, 46000, 50000, 59000, 68000, 73000, 86000]  # 目的変数
#
# df = pd.DataFrame({
#     'area': area,
#     'age': age,
#     'rent': rent})
# データを可視化してみる
# fig = px.scatter_3d(df, x='area', y='age', z='rent', color='rent', opacity=0.7)
# fig.update_traces(marker=dict(
#     size=8,
#     line=dict(width=2,color='white')))
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# fig.show()

df_base = pd.read_csv("../data/house-prices-advanced-regression-techniques/train.csv")
df = df_base.loc[:, ["GrLivArea", "YearBuilt", "SalePrice"]]
feature_names = ["GrLivArea", "YearBuilt"]
y_name = "SalePrice"
X = df_base.loc[:, feature_names].values
y = df_base[y_name].values
ones = np.ones(len(X)).reshape(-1, 1)
X = np.hstack((ones, X))

# 行列X, ベクトルyの準備
X = df[['GrLivArea', 'YearBuilt']].values
ones = np.ones(len(X)).reshape(-1, 1)
X = np.hstack((ones, X))
print('行列X')
print(X)
y = df['SalePrice'].values
print('ベクトルy')
print(y)

def multiple_regression(X, y):
    """回帰係数ベクトルを計算する"""
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta

theta = multiple_regression(X, y)
theta_0 = theta[0]
theta_1 = theta[1]
theta_2 = theta[2]

print(f'theta_0: {theta_0}')
print(f'theta_1: {theta_1}')
print(f'theta_2: {theta_2}')


# 回帰平面の可視化

mesh_size = 1
margin = 0.1
x1_min, x1_max = df['GrLivArea'].min()-margin, df['GrLivArea'].max()+margin
x2_min, x2_max = df['YearBuilt'].min()-margin, df['YearBuilt'].max()+margin
x1_range = np.arange(x1_min, x1_max, mesh_size)
x2_range = np.arange(x2_min, x2_max, mesh_size)
xx1, xx2 = np.meshgrid(x1_range, x2_range)

y = (theta_0 + theta_1 * xx1 + theta_2 * xx2)  # 回帰平面

fig = px.scatter_3d(df, x='GrLivArea', y='YearBuilt', z='SalePrice', opacity=0.8)  # データ点のプロット
fig.update_traces(marker=dict(
    color=colors[0],
    size=8,
    line=dict(width=2,color='white')))

fig.add_traces(go.Surface(x=x1_range, y=x2_range, z=y, opacity=0.7))  # 平面のプロット
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
pio.write_html(fig, 'multiple-regression_fig2.html', include_plotlyjs='cdn', full_html=False)  # グラフを保存

