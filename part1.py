import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt
import helpers as hlp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def p1_plots(q, dataSet):
    fin = open("{}.pkl".format(dataSet), "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test
    X_te_ones = np.array([X_te[i] for i in range(len(X_te)) if y_te[i]==1])
    X_te_nils = np.array([X_te[i] for i in range(len(X_te)) if y_te[i]==0])
    X_tr_ones = np.array([X_tr[i] for i in range(len(X_tr)) if y_tr[i]==1])
    X_tr_nils = np.array([X_tr[i] for i in range(len(X_tr)) if y_tr[i]==0])
    
    k = []
    if(q == 1):
        k = [1,2,3,4,5]
    elif(q == 2):
        k = [1,2,3,4,None]
    elif(q == 3):
        k = ["linear", "rbf", "poly"]

    for i in k:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if(q == 1):
            ax.set_title("n_neighbors = %d" % (i))
        elif(q == 2):
            if(i is not None):
                ax.set_title("max_depth = %d" % (i))
            else:
                ax.set_title("max_depth = None")
        elif(q == 3):
            ax.set_title("kernel = %s" % (i))
        ax.scatter(X_tr_ones[:,0], X_tr_ones[:,1], c='b')
        ax.scatter(X_tr_nils[:,0], X_tr_nils[:,1], c='r')
        ax.scatter(X_te_ones[:,0], X_te_ones[:,1],c='b', marker = "*")
        ax.scatter(X_te_nils[:,0], X_te_nils[:,1],c='r', marker = "*")
        if(q == 1):
            clf = KNeighborsClassifier(n_neighbors = i)
        elif(q == 2):
            clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
        elif(q == 3):
            clf = SVC(kernel = i)
        clf.fit(X_tr, y_tr)
        y_tr_pred = clf.predict(X_tr)
        y_te_pred = clf.predict(X_te)
        tr_loss = len([y_tr[i] for i in range(len(y_tr)) if y_tr[i]!=y_tr_pred[i]]) / len(y_tr)
        te_loss = len([y_te[i] for i in range(len(y_te)) if y_te[i]!=y_te_pred[i]]) / len(y_te)
        ax.set_xlabel("Tr Loss: {} \n Te Loss: {}".format(tr_loss, te_loss))
        x_min, x_max, y_min, y_max = hlp.get_bounds(X_tr)
        hlp.plot_decision_boundary(ax, clf, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        #plt.savefig("knn_plot_n%d.pdf" % (n))
        # or if that takes too long: plot_decision_boundary(ax, clf, res=.1)
    plt.show()

if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    print(f"Version of sklearn: {sklearn.__version__}")
    print("(It should be 0.20.0)")

    p1_plots(1, "simple_task")
    p1_plots(1, "moons")
    p1_plots(2, "simple_task")
    p1_plots(2, "moons")
    p1_plots(3, "simple_task")
    p1_plots(3, "moons")
    

    
