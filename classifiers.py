from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression 

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Used to ignore warnings generated from StackingCVClassifier
import warnings
warnings.simplefilter('ignore')

class MADA:
    
    def __init__(self, args):
        super(MADA, self).__init__()

        self.args = args

        self.num_classifiers = 6
        
        classifier1 = GaussianProcessClassifier(1.0 * RBF(1.0), max_iter_predict=50, random_state = 0)

        classifier2 = MLPClassifier(hidden_layer_sizes=(100),alpha=1, max_iter=1000, random_state = 0)
    
        classifier3 = SVC(kernel="linear", C=0.025, probability=True, random_state = 0)

        classifier4 =  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state = 0)

        classifier5 = LogisticRegression(random_state = 0)
        
        sclf = StackingCVClassifier(classifiers = [classifier1, classifier2, classifier3, classifier4, classifier5],
                                    shuffle = False,
                                    use_probas = True,
                                    cv = 5,
                                    random_state=0,
                                    meta_classifier = LogisticRegression())

        self.classifiers = {"GPC": classifier1,
                            "MLP": classifier2,
                            "SVC": classifier3,
                            "RF": classifier4,
                            "LR": classifier5,
                            "Stack": sclf}

    
    def train(self, X_train, y_train, X_test, y_test):
        
        arr = np.empty((X_test.shape[0], self.num_classifiers))
        
        y_pred = np.empty((X_test.shape[0],1))
        
        ind = 0

        for key in self.classifiers:
            print(f"\n=== Training ===: {key}")
            print(key, end=" : ")
            # Get classifier
            classifier = self.classifiers[key]
            # Fit classifier
            classifier.fit(X_train, y_train)
            # Save fitted classifier
            self.classifiers[key] = classifier
            y_pred = self.classifiers[key].predict(X_test)
            print(accuracy_score(y_test, y_pred))
            arr[:,ind] = y_pred
            ind += 1
        
        print("\n=== Voting ===")
        for i in range(X_test.shape[0]):
            y_pred[i]=self.findMajority(arr[i,:], self.num_classifiers)
        
        print("Majority voting: ", accuracy_score(y_test, y_pred))

    
    def findMajority(self, arr, n):
        maxCount = 0
        index = -1  # sentinels
        for i in range(n):
            count = 0
            for j in range(n):
                if(arr[i] == arr[j]):
                    count += 1
            if(count > maxCount):
                maxCount = count
                index = i
        return arr[index]
