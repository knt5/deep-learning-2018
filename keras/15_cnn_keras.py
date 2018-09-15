import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(123)

'''
データの生成
'''
mnist = datasets.fetch_mldata('MNIST original', data_home='..')

n = len(mnist.data)
N = 30000  # MNISTの一部を使う
N_train = 20000
N_validation = 4000
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

X = mnist.data[indices]
X = X / 255.0                # 正規化 (0～1化)
X = X - X.mean(axis=1).reshape(len(X), 1)  # 平均を引いている
X = X.reshape(N, 28, 28, 1)  # MNISTの1次元画像データを2次元画像データに変換。
                             # カラー画像では X.reshape(N, 28, 28, 3) ※最後が1ではなく3。
                             # RGB。RGBになってもフィルタ6枚用意したら、畳み込みで6枚の画像しか
                             # 生成されない。R,G,Bが畳み込まれてR',G',B'が出来て、それの総和が
                             # 特徴マップとして得られる。
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, train_size=N_train)
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_train, Y_train, test_size=N_validation)

'''
モデル設定
'''
n_in = len(X[0])  # 784
n_out = len(Y[0])  # 10


def weight_variable(shape, name=None):
    return np.sqrt(2.0 / shape[0]) * np.random.normal(size=shape)


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model = Sequential()

# 1st Convolution Layer
model.add(Conv2D(32, (5, 5), padding='same',  # 32枚
                 data_format='channels_last',  # データによっては channels_first。チャンネル情報が最初にある場合。
                 input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolution Layer
model.add(Conv2D(64, (5, 5), padding='same'))  # 64枚
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layer
model.add(Flatten())  # 3次元を1次元にする
                      # ★ 最近はFlatten しない。平均値プーリング取ってそのまま softmax に入力。= Global Average Pooling
model.add(Dense(1024))         # つまりこの隠れ層もナシ
model.add(Activation('relu'))  # ナシ
model.add(Dropout(0.5))        # ナシ

# Readout Layer
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999),  # Adam 使用
              metrics=['accuracy'])

'''
モデル学習
'''
epochs = 200
batch_size = 200

hist = model.fit(X_train, Y_train, epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_validation, Y_validation),
                 callbacks=[early_stopping])

'''
学習の進み具合を可視化
'''
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(len(val_loss)), val_loss, label='loss', color='black')
plt.xlabel('epochs')
plt.show()

'''
予測精度の評価
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
