import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import SGD


def create_net():
    model = Sequential()

    # model.add(Dense(140, input_dim=140, init='glorot_uniform'))
    # model.add(Activation('sigmoid'))
    model.add(Dense(400, input_dim=140, init='he_normal'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(250, init='he_normal'))
    model.add(Activation('sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #
    # model.add(Dense(50, init = 'he_normal'))
    # model.add(PReLU())
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Dense(29, init='uniform'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0, nesterov=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    return model

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 5)


df = pd.read_csv('sfpd_nn.csv')
df.fillna(0, inplace=True)
y = pd.get_dummies(df['Category']).values
X = df.drop(['Category'], axis=1).values

# evaluate using 10-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=2017)
score = 0
for idx_tr, idx_test in kfold.split(X, y):
    # print idx_tr
    X_train, y_train = X[idx_tr], y[idx_tr]
    X_test, y_test = X[idx_test], y[idx_test]
    model = create_net()
    model.fit(X_train, y_train, nb_epoch=20, batch_size=2000, verbose=0)
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print '_'*30
    print("train %s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("train %s: %.2f%%" % (model.metrics_names[2], scores_train[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print
    score += scores[2]
score /= 10
print score * 100
    # # cvscores.append(scores[1] * 100)
