import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def get_favorite_data():
    y = np.random.binomial(1, 0.5) #flip a coin for y

    d = 2

    if y == 0:
        a = (random.uniform(0,30), random.uniform(11,15.2))
        b = (random.uniform(0,30), random.uniform(16,28.2))
        c = (random.uniform(0,30), random.uniform(29,30))
        d = (random.uniform(0,30), random.uniform(0,10.2))
        choices = [a,b,c,d]
        x = random.choice(choices)
    else:
        a = (random.uniform(0,30), random.uniform(15,16))
        b = (random.uniform(0,30), random.uniform(10,11))
        c = (random.uniform(0,30), random.uniform(28,29))
        choices = [a,b,c]
        x = random.choice(choices) 

    return x, y

def example_get_favorite_data():
    # Two, far apart, spherical Gaussian blobs
    d = 1
    
    mu0 = np.array([-1 for i in range(d)])
    mu1 = np.array([ 1 for i in range(d)])

    y = np.random.binomial(1, 0.5) #flip a coin for y

    if y == 0:
        x = np.random.multivariate_normal(mean = mu0, cov = np.eye(d))
    else:
        x = np.random.multivariate_normal(mean = mu1, cov = np.eye(d))

    return x, y

def get_lots_of_favorite_data(n = 100, data_fun = get_favorite_data):
    pts = [data_fun() for _ in range(n)]
    Xs, ys = zip(*pts)
    X = np.array(Xs)
    y = np.array(ys)
    return (X, y)

if __name__ == "__main__":
    """
    print("Here are some points from example_get_favorite_data:")
    for i in range(4):
        x, y = example_get_favorite_data()
        print(f"\tx: {x}")
        print(f"\ty: {y}")

    print("And here we use get_lots_of_favorite_data to obtain X and y:")
    X, y = get_lots_of_favorite_data(10, example_get_favorite_data)

    print("X:")
    print(X)
    print("y:")
    print(y)"""


    X, y = get_lots_of_favorite_data(n= 100, data_fun=get_favorite_data)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = .25, random_state = 69)

    knn1 = KNeighborsClassifier(n_neighbors = 1)
    knn2 = KNeighborsClassifier(n_neighbors = 2)
    knn3 = KNeighborsClassifier(n_neighbors = 3)
    knn4 = KNeighborsClassifier(n_neighbors = 4)
    knn5 = KNeighborsClassifier(n_neighbors = 5)

    knn1.fit(X_tr, y_tr)
    knn2.fit(X_tr, y_tr)
    knn3.fit(X_tr, y_tr)
    knn4.fit(X_tr, y_tr)
    knn5.fit(X_tr, y_tr)

    knn1_preds = knn1.predict(X_te)
    knn2_preds = knn2.predict(X_te)
    knn3_preds = knn3.predict(X_te)
    knn4_preds = knn4.predict(X_te)
    knn5_preds = knn5.predict(X_te)

    knn1_loss = zero_one_loss(knn1_preds, y_te)
    knn2_loss = zero_one_loss(knn2_preds, y_te)
    knn3_loss = zero_one_loss(knn3_preds, y_te)
    knn4_loss = zero_one_loss(knn4_preds, y_te)
    knn5_loss = zero_one_loss(knn5_preds, y_te)

    print("knns:")
    print(knn1_loss)
    print(knn2_loss)
    print(knn3_loss)
    print(knn4_loss)
    print(knn5_loss)

    tree1 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1)
    tree2 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)
    tree3 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
    tree4 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
    Laura = DecisionTreeClassifier(criterion = 'entropy', max_depth = None)

    tree1.fit(X_tr, y_tr)
    tree2.fit(X_tr, y_tr)
    tree3.fit(X_tr, y_tr)
    tree4.fit(X_tr, y_tr)
    Laura.fit(X_tr, y_tr)

    tree1_preds = tree1.predict(X_te)
    tree2_preds = tree2.predict(X_te)
    tree3_preds = tree3.predict(X_te)
    tree4_preds = tree4.predict(X_te)
    Laura_preds = Laura.predict(X_te)

    tree1_loss = zero_one_loss(tree1_preds, y_te)
    tree2_loss = zero_one_loss(tree2_preds, y_te)
    tree3_loss = zero_one_loss(tree3_preds, y_te)
    tree4_loss = zero_one_loss(tree4_preds, y_te)
    Laura_loss = zero_one_loss(Laura_preds, y_te)

    print("trees:")
    print(tree1_loss)
    print(tree2_loss)
    print(tree3_loss)
    print(tree4_loss)
    print("Laura Loss: ")
    print(Laura_loss)

    svmLin = SVC(kernel = 'linear')
    svmRBF = SVC(kernel = 'rbf')
    svmPoly = SVC(kernel = 'poly')

    svmLin.fit(X_tr, y_tr)
    svmRBF.fit(X_tr, y_tr)
    svmPoly.fit(X_tr, y_tr)

    svmLin_preds = svmLin.predict(X_te)
    svmRBF_preds = svmRBF.predict(X_te)
    svmPoly_preds = svmPoly.predict(X_te)

    svmLin_loss = zero_one_loss(svmLin_preds, y_te)
    svmRBF_loss = zero_one_loss(svmRBF_preds, y_te)
    svmPoly_loss = zero_one_loss(svmPoly_preds, y_te)

    print("SVMs:")
    print(svmLin_loss)
    print(svmRBF_loss)
    print(svmPoly_loss)

    losses = [knn1_loss, knn2_loss, knn3_loss, knn4_loss, knn5_loss, tree1_loss, tree2_loss, tree3_loss, tree4_loss, Laura_loss, svmLin_loss, svmRBF_loss, svmPoly_loss]

    #losses = [knn1_loss, knn2_loss, knn3_loss, knn4_loss, knn5_loss, tree1_loss, tree2_loss, tree3_loss, tree4_loss, treeN_loss, svmLin_loss, svmRBF_loss]

    minLoss = np.argmin(losses)
    print("MinLoss: {}".format(minLoss))
    
    if(minLoss == 0):
        print("knn1")
    elif(minLoss == 1):
        print("knn2")
    elif(minLoss == 2):
        print("knn3")
    elif(minLoss == 3):
        print("knn4")
    elif(minLoss == 4):
        print("knn5")
    elif(minLoss == 5):
        print("tree1")
    elif(minLoss == 6):
        print("tree2")
    elif(minLoss == 7):
        print("tree3")
    elif(minLoss == 8):
        print("tree4")
    elif(minLoss == 9):
        print("Laura wins!")
    elif(minLoss == 10):
        print("svmLin")
    elif(minLoss == 11):
        print("svmRBF")
    elif(minLoss == 12):
        print("svmPoly")



