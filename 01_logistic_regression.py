import numpy as np

'''
データの生成
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 入力
Y = np.array([[0], [1], [1], [1]])              # 出力

'''
モデル設定
'''
w = np.zeros(2)      # 重み
b = .0               # バイアス (+b, 元 -Θ)
learning_rate = 0.1  # 学習率エータ(η)

# ロジスティック回帰の実装
def y(x):
    return sigmoid(np.dot(w, x) + b)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

'''
モデル学習
'''
for epoch in range(20000):   # 反復番号の k を epoch と呼ぶ
    delta_w = .0
    delta_b = .0

    for i in range(len(X)):
        preds = y(X[i])   # 予測値
        t = Y[i][0]       # 実際の値

        delta_w += (t - preds) * X[i]  # (t - y) * x
        delta_b += t - preds           # t - y

    w += learning_rate * delta_w
    b += learning_rate * delta_b

'''
学習結果の確認
'''
print('output probability:')
for _, x in enumerate(X):
    print(y(x))
