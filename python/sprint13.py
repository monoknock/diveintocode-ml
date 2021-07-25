#
# Sequentialクラス パターン
#
import numpy as np

# ANDゲートの訓練データを用意
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[0],[0],[1]])

import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation = tf.nn.sigmoid, input_shape=(2,))])

model.summary()

# コンパイル
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    batch_size=1,
    epochs=1000,
    verbose=0)

y_pred_proba = model.predict(x_train)[:, 0]
# 確率を0, 1に変換
y_pred = np.where(y_pred_proba >0.5, 1, 0)
print("y_pred_proba", y_pred_proba)
print("y_pred", y_pred)

score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])



#
# Functional API
#

input_data = tf.keras.layers.Input(shape=(2,)) # 入力層
output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(input_data) # 出力層

model = tf.keras.Model(inputs=input_data, outputs=output)

model.summary()
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy'])
history = model.fit(
    x_train, y_train,
    batch_size=1,
    epochs=1000,
    verbose=0)

