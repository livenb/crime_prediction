import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc, f1_score
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import re


def load_data(filename):
    return pd.read_csv(filename)


def get_rid_num(x):
    return re.sub(r'\d+', '', x)


def data_preprocess(df):
    dropLst = ['Unnamed: 0', 'STATION_NAME',
               'STATISTICAL_CODE_DESCRIPTION', 'CrimeCat']
    df['STREET'] = df['STREET'].apply(get_rid_num)
    df['ZIP'] = df['ZIP'].apply(int)
    le = LabelEncoder()
    df['STREET'] = le.fit_transform(df['STREET'])
    df['CITY'] = le.fit_transform(df['CITY'])
    X = df.drop(dropLst, axis=1).values
    y = df['CrimeCat'].values
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    return X, y


def build_one_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                oob_score=True, n_jobs=-1, verbose=1)
    ovr = OneVsRestClassifier(estimator=rf, n_jobs=-1)
    ovr.fit(X_train, y_train)
    y_score = ovr.predict_proba(X_test)
    return ovr, y_score



def build_grid_search(X, y):
    parameters = {
        "estimator__criterion": ['gini', 'entropy'],
        "estimator__max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    model_tunning = GridSearchCV(model_to_set, param_grid=parameters,
                                 score_func=make)
    model_tunning.fit(X, y)

def multiclass_roc(y_score, n_classes=10):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('img/roc_subsample.png')
    plt.show()

if __name__ == '__main__':
    filename = 'data/la_clean.csv'
    df = load_data(filename)
    sample = df.sample(frac=0.3)
    X, y = data_preprocess(sample)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model, y_score = build_one_model(X_train, y_train, X_test, y_test)
    score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    # print score, test_score
    multiclass_roc(y_score)
