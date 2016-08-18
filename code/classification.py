# import  matplotlib
# matplotlib.use('Agg')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import roc_auc_score
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import re


crimes = {1:'Theft/Larcery', 2:'Robebery', 3:'Nacotic/Alcochol',
          4:'Assault', 5:'Grand Auto Theft', 6: 'Vandalism',
          7:'Burglary', 8:'Homicide', 9:'Sex Crime', 10:'DUI'}


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
    feature_names = df.drop(dropLst, axis=1).columns
    X = df.drop(dropLst, axis=1).values
    y = df['CrimeCat'].values
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    return X, y, feature_names


def build_one_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=1000, criterion='gini',
                                oob_score=True, n_jobs=-1, verbose=1)
    # ovr = OneVsRestClassifier(estimator=rf, n_jobs=-1)
    ovr = OneVsRestClassifier(estimator=rf, n_jobs=-1)
    ovr.fit(X_train, y_train)
    y_score = ovr.predict_proba(X_test)
    return ovr, y_score


def get_feature_importance(models, fea_names):
    fea_imp = []
    tops = []
    for model in models:
        fea = model.feature_importances_
        idx = np.argsort(fea)[::-1]
        fea_imp.append(fea)
        tops.append(fea_names[idx][:3])
        print fea_names[idx][:3]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(fea)), fea[idx], align="center")
        plt.xticks(range(len(fea)), fea_names[idx], rotation=30, alpha=0.7)
        plt.xlim([-1, len(fea)])
    plt.show()
    return fea_imp, tops


def plot_fea_impor(fea, fea_names, idx):
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(fea)), fea[idx],
            color="r", align="center")
    plt.xticks(range(len(fea)), fea_names[idx])
    plt.xlim([-1, len(fea)])
    # plt.show()


def build_grid_search(X, y):
    parameters = {
        "estimator__criterion": ['gini', 'entropy'],
        "estimator__max_depth": [10, 15, 20, 25, None],
        "estimator__max_features": ['auto', 'sqrt', 'log2', None]
    }
    ovr = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000,
                                    oob_score=True, n_jobs=-1, verbose=1))
    model_tunning = GridSearchCV(ovr, param_grid=parameters, verbose=1,
                                 n_jobs=-1, cv=10,
                                 scoring=make_scorer(f1_score))
    model_tunning.fit(X, y)
    test_score = model_tunning.best_score_
    print 'The best test score: ', test_score
    y_score = model_tunning.predict_proba(X_test)
    multiclass_roc(y_score, 'grid_search_02')
    return model_tunning


def multiclass_roc(y_score, title=None, n_classes=10):
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
    plt.figure(figsize=(15,15))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of {0} (area = {1:0.2f})'
                                       ''.format(crimes[i+1], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    if title != None:
        plt.savefig('img/'+title+'.png')
    plt.show()

if __name__ == '__main__':
    filename = 'data/la_clean.csv'
    df = load_data(filename)
    sample = df.sample(frac=0.1)
    X, y, fea_names = data_preprocess(sample)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model, y_score = build_one_model(X_train, y_train, X_test, y_test)
    # print score, test_score
    multiclass_roc(y_score, )
    fea_import, tops = get_feature_importance(model.estimators_, fea_names)
    # gvmodel = build_grid_search(X_train, y_train)
    
