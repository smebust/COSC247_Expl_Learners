import numpy as np
import dill as pickle
import sklearn
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as mets
import helpers as hlp
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def find_numnum(X_tr, y_tr):
    nils = 0
    ones = 0
    twos = 0
    threes = 0
    fours = 0
    fives = 0
    sixes = 0
    sevens = 0
    eights = 0
    nines = 0
    for i in range(10):
        toget = [X_tr[n] for n in range(len(X_tr)) if y_tr[n] == i]
        if(i == 0):
            nils = len(toget)
        elif(i == 1):
            ones = len(toget)
        elif(i == 2):
            twos = len(toget)
        elif(i == 3):
            threes = len(toget)
        elif(i == 4):
            fours = len(toget)
        elif(i == 5):
            fives = len(toget)
        elif(i == 6):
            sixes = len(toget)
        elif(i == 7):
            sevens = len(toget)
        elif(i == 8):
            eights = len(toget)
        elif(i == 9):
            nines = len(toget)
    print("Zeros: %d" % nils)
    print("Ones: %d" % ones)
    print("Twos: %d" % twos)
    print("Threes: %d" % threes)
    print("Fours: %d" % fours)
    print("Fives: %d" % fives)
    print("Sixes: %d" % sixes)
    print("Sevens: %d" % sevens)
    print("Eights: %d" % eights)
    print("Nines: %d" % nines)

def get_Q1_2():
    print("\nQuestions 1 and 2:\n")
    fin = open("digits.pkl", "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test
    find_numnum(X_tr, y_tr)

    clf = LogisticRegression()

    clf.fit(X_tr, y_tr)
    y_preds = clf.predict(X_te)


    for i in range(10):
        toav_Xtr = np.array([X_tr[n] for n in range(len(X_tr)) if y_tr[n] == i])
        toplt_tr = np.array([np.mean(toav_Xtr[:,k]) for k in range(toav_Xtr.shape[1])])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        hlp.plot_num(ax, toplt_tr)
        plt.show()

        wrong = []
        for r in range(len(X_te)):
            if(y_te[r] == i and not y_preds[r] == i):
                wrong = X_te[r]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        hlp.plot_num(ax, wrong)
        plt.show()

        prec = mets.precision_score(y_te, y_preds, labels = [i], average = 'macro')
        rec = mets.recall_score(y_te, y_preds, labels = [i], average = 'macro')

        print("Precision for {}: {}".format(i, prec))
        print("Recall for {}: {}".format(i, rec))
    cmatr = mets.confusion_matrix(y_te, y_preds, labels = [0,1,2,3,4,5,6,7,8,9])
    print("Confusion Matrix:")
    print(cmatr)

def get_Q3(scaled = False):
    if(scaled):
        print("\nQ3 with scaled data:\n")
    else:
        print("\nQ3:\n")
    fin = open("cancer.pkl", "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    clf = KNeighborsClassifier(n_neighbors = 3)

    if(scaled):
        scl = MinMaxScaler()
        new_Xtr = scl.fit_transform(X_tr)
        #print("new_Xtr: {}".format(new_Xtr))
        clf.fit(new_Xtr, y_tr)
        new_Xte = scl.fit_transform(X_te)
        canc_preds = clf.predict(new_Xte)
        cmatr = mets.confusion_matrix(y_te, canc_preds)
        print("Confusion Matrix: ")
        print(cmatr)
    else:
        #print("X_tr: {}".format(X_tr))
        clf.fit(X_tr, y_tr)
        canc_preds = clf.predict(X_te)
        cmatr = mets.confusion_matrix(y_te, canc_preds)
        print("Confusion Matrix: ")
        print(cmatr)

def get_Q4():
    print("\nQ4:\n")
    fin = open("cancer.pkl", "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    scl = MinMaxScaler()
    new_Xtr = scl.fit_transform(X_tr)

    param_grid = [{'C': [1,10,100,1000], 'gamma': [1, .1, .01, .001], 'kernel': ['rbf']},
    {'C': [1,10,100,1000], 'degree': [2, 3, 4, 5], 'kernel': ['poly']}, 
    {'C': [1,10,100,1000], 'coef0': [0.1,1,10,100], 'kernel': ['sigmoid']}]

    svc = SVC()
    clf = GridSearchCV(svc, param_grid, cv=5, scoring = 'f1')

    clf.fit(new_Xtr, y_tr)

    new_Xte = scl.fit_transform(X_te)
    y_preds = clf.predict(new_Xte)

    cmatr = mets.confusion_matrix(y_te, y_preds)
    print("Confusion Matrix:")
    print(cmatr)

    f1score = mets.f1_score(y_te, y_preds)

    prec = mets.precision_score(y_te, y_preds, average = 'micro')
    rec = mets.recall_score(y_te, y_preds, average = 'micro')

    print("Precision: {} \nRecall: {} \nF1 Score: {}".format(prec, rec, f1score))

    hparams = clf.best_estimator_
    print("Parameters: ")
    print(hparams)

if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    print(f"Version of sklearn: {sklearn.__version__}")
    print("(It should be 0.20.0)")

    get_Q1_2()

    get_Q3()
    get_Q3(scaled=True)

    get_Q4()

    











    
