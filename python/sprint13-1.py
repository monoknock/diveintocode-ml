import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# PrefetchDataset
mnist_dataset, mnist_info = tfds.load(name="mnist", with_info=True, as_supervised=True)
# PrefetchDataset
mnist_train, mnist_test = mnist_dataset["train"], mnist_dataset['test']

# EagerTensor
# なぜtf.castを通しているのかは現段階では不明、おそらくtf内で計算しやすいオブジェクトにしているのではなかろうか
num_valid_samples = tf.cast(0.1 * mnist_info.splits['train'].num_examples, tf.int64)
num_test_samples = tf.cast(mnist_info.splits['test'].num_examples, tf.int64)


# Feature Scaling 0~255 => 0~1 にスケーリングする
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label


# MapDataset
scaled_train_and_valid_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

# train/validにsplit
BUFFER_SIZE = 10000 # メモリ対策
shuffled_train_and_valid_data = scaled_train_and_valid_data.shuffle(BUFFER_SIZE)
valid_data = shuffled_train_and_valid_data.take(num_valid_samples)
train_data = shuffled_train_and_valid_data.skip(num_valid_samples)

# 負荷対策
BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)
valid_data = valid_data.batch(num_valid_samples)
test_data = test_data.batch(num_test_samples)

valid_inputs, validation_targets = next(iter(valid_data))