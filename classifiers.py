from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import cPickle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

__author__ = "Mary Ziemba, based on code by Mark Nemececk for COMPSCI 270, Spring 2017, Duke University"
__copyright__ = "Mary Ziemba"
__credits__ = ["Mary Ziemba", "Alex Deckey", "David Duquette",
                    "Camila Vargas Restrepo", "Melanie Krassel"]
__license__ = "Creative Commons Attribution-NonCommercial 4.0 International License"
__version__ = "1.0.0"
__email__ = "mtz3@duke.edu"

def create_classifier(model_type='decision_tree', random_model=False):
    model = None
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(max_features=None,random_state=0)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=10)
    elif model_type == 'gaussian_nb':
        model = GaussianNB()
    elif model_type == 'random_forest':
        if random_model:
            model = RandomForestClassifier(n_estimators=20)
    return model

def extract_features(data, indices):
    return data[:, indices]


def calculate_model_accuracy(predict_train, predict_test, target_train, target_test):
    train_accuracy = metrics.accuracy_score(target_train, predict_train)
    test_accuracy = metrics.accuracy_score(target_test, predict_test)
    return (train_accuracy, test_accuracy)

def calculate_confusion_matrix(predict_test,target_test):
    return metrics.confusion_matrix(target_test,predict_test)

def create_decision_tree(max_features=None, random_state=0):
    '''Returns a sklearn.DecisionTreeClassifier with the max_features provided'''
    model = DecisionTreeClassifier(max_features=max_features, random_state=random_state)
    return model


def create_random_forest(n_estimators=10, random_state=0):
    '''Returns a sklearn.RandomForestClassifier with the n_estimators provided'''
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    return model