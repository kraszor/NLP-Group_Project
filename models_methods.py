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


class ModelsTraining:
    def __init__(self, train_data, val_data) -> None:
        self.train_x = train_data[0]
        self.train_y = train_data[1]
        self.val_x = val_data[0]
        self.val_y = val_data[1]
        self.logistic_regression = LogisticRegression(random_state=42)
        self.nb = MultinomialNB(random_state=42)
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.knn = KNeighborsClassifier(random_state=42)
        self.random_forest = RandomForestClassifier(random_state=42)

    def grid_search_fit(self, param_grid, model, metric):
        grid_search = GridSearchCV(model, param_grid, cv=5,
                                   scoring=metric,
                                   return_train_score=False)
        grid_search.fit(self.train_x, self.train_y)
        return grid_search.best_estimator_

    def train_logistic_reg(self, param_grid):
        model = self.grid_search_fit(param_grid,
                                     self.logistic_regression,
                                     'f1')
        return model

    def train_decision_tree(self, param_grid):
        model = self.grid_search_fit(param_grid, self.decision_tree, 'f1')
        return model

    def train_nb(self, param_grid):
        model = self.grid_search_fit(param_grid, self.nb, 'f1')
        return model

    def train_knn(self, param_grid):
        model = self.grid_search_fit(param_grid, self.knn, 'f1')
        return model

    def train_random_forest(self, param_grid):
        model = self.grid_search_fit(param_grid, self.random_forest, 'f1')
        return model

    def train_neural_network(self):
        pass


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


class ModelEval:
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
