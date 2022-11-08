import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (f1_score, accuracy_score,
                             roc_curve, roc_auc_score,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV


PARAM_GRID = {'logistic_regression': {"C": np.logspace(-3, 3, 7),
                                      "penalty": ["l1", "l2"]},
              'decision_tree': {'criterion': ['gini', 'entropy'],
                                'max_depth': [4, 5, 6, 7, 8, 9,
                                              10, 11, 12, 15, 20,
                                              30, 40, 50, 70, 90,
                                              120, 150]},
              'naive_bayes': {'alpha': [0.00001, 0.0001, 0.001,
                                        0.1, 1, 10, 100, 1000]},
              'k_nearest_neighbors': {'n_neigbors': list(range(1, 31))},
              'random_forest': {'n_estimators': [200, 500],
                                'max_features': ['auto', 'sqrt', 'log2'],
                                'max_depth': [4, 5, 6, 7, 8],
                                'criterion': ['gini', 'entropy']}}


class ModelsTraining:
    def __init__(self, train_data, val_data=(None, None)) -> None:
        self.train_x = train_data[0]
        self.train_y = train_data[1]
        self.val_x = val_data[0]
        self.val_y = val_data[1]
        self.__logistic_regression = LogisticRegression(random_state=42)
        self.__nb = MultinomialNB()
        self.__decision_tree = DecisionTreeClassifier(random_state=42)
        self.__knn = KNeighborsClassifier()
        self.__random_forest = RandomForestClassifier(random_state=42)

    def grid_search_fit(self, param_grid, model, metric):
        grid_search = GridSearchCV(model, param_grid, cv=5,
                                   scoring=metric,
                                   return_train_score=False)
        grid_search.fit(self.train_x, self.train_y)
        return grid_search.best_estimator_

    def train_models(self, param_grid):
        logistic_reg = self.grid_search_fit(param_grid['logistic_regression'],
                                            self.__logistic_regression,
                                            'f1')
        decision_tree = self.grid_search_fit(param_grid['decision_tree'],
                                             self.__decision_tree, 'f1')
        nb = self.grid_search_fit(param_grid['naive_bayes'], self.__nb, 'f1')
        knn = self.grid_search_fit(param_grid['k_nearest_neighbors'],
                                   self.__knn, 'f1')
        random_forest = self.grid_search_fit(param_grid['random_forest'],
                                             self.__random_forest, 'f1')
        neural_network = None
        models = {'logistic_reg': logistic_reg,
                  'decision_tree': decision_tree,
                  'naive_bayes': nb,
                  'knn': knn,
                  'random_forest': random_forest,
                  'neural_network': neural_network}
        return models


class ModelsPredict:
    def __init__(self, test_set, train_set) -> None:
        self.test_x = test_set[0]
        self.test_y = test_set[1]
        self.train_x = train_set[0]
        self.train_y = train_set[1]

    def predict_train(self, model):
        if model != 'neural_network':
            y_hat = model.predict(self.train_x)
        else:
            y_hat = model(self.train_x)
        return y_hat

    def predict_test(self, model):
        if model != 'neural_network':
            y_hat = model.predict(self.test_x)
        else:
            y_hat = model(self.test_x)
        return y_hat


class ModelsEval:
    def __init__(self) -> None:
        pass

    def eval_metrics(self, y, y_hat):
        accuracy = f"Accuracy: {accuracy_score(y, y_hat)}\n"
        f1 = f"F1 Score: {f1_score(y, y_hat)}\n"
        curve = roc_curve(y, y_hat)
        roc_score = f"ROC AUC Score: {roc_auc_score(y, y_hat)}\n"
        report = classification_report(y, y_hat)
        return accuracy + f1 + roc_score, curve, report

    def conf_matrix(self, y, y_hat):
        cm = confusion_matrix(y, y_hat)
        disp = ConfusionMatrixDisplay(cm)
        return disp  # next disp.plot() disp.show()
