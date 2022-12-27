import numpy as np
import pandas as pd
import pickle
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (f1_score, accuracy_score,
                             roc_curve, roc_auc_score,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


PARAM_GRID = {'logistic_regression': {"C": np.logspace(-3, 3, 7),
                                      "penalty": ["l1", "l2"]},
              'decision_tree': {'criterion': ['gini', 'entropy'],
                                'max_depth': [4, 5, 6, 7, 8, 9,
                                              10, 11, 12, 15, 20,
                                              30, 40, 50, 70, 90,
                                              120, 150]},
              'naive_bayes': {'alpha': [0.00001, 0.0001, 0.001,
                                        0.1, 1, 10, 100, 1000]},
              'k_nearest_neighbors': {'n_neighbors': list(range(1, 31))},
              'random_forest': {'n_estimators': [200, 500],
                                'max_features': ['auto', 'sqrt', 'log2'],
                                'max_depth': [4, 5, 6, 7, 8],
                                'criterion': ['gini', 'entropy']},
              'svc': {'C': [0.1, 0.5, 1, 10, 100, 1000],
                      'penalty': ('l1', 'l2'),
                      'loss': ('hinge', 'squared_hinge')}}


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
        self.__svc = LinearSVC(random_state=42)

    def grid_search_fit(self, param_grid, model, metric):
        if not os.path.isfile(f'models/{model}.pkl'):
            grid_search = GridSearchCV(model, param_grid, cv=5,
                                       scoring=metric,
                                       return_train_score=False)
            grid_search.fit(self.train_x, self.train_y)
            pickle.dump(grid_search.best_estimator_,
                        open(f'models/{model}.pkl', 'wb'))

    def train_models(self, param_grid):
        self.grid_search_fit(param_grid['logistic_regression'],
                             self.__logistic_regression,
                             'f1')
        self.grid_search_fit(param_grid['decision_tree'],
                             self.__decision_tree, 'f1')
        self.grid_search_fit(param_grid['naive_bayes'], self.__nb, 'f1')
        self.grid_search_fit(param_grid['k_nearest_neighbors'],
                             self.__knn, 'f1')
        self.grid_search_fit(param_grid['random_forest'],
                             self.__random_forest, 'f1')
        self.grid_search_fit(param_grid['svc'],
                             self.__svc, 'f1')
        models = (f'{self.__logistic_regression}',
                  f'{self.__decision_tree}',
                  f'{self.__nb}',
                  f'{self.__knn}',
                  f'{self.__random_forest}',
                  f'{self.__svc}')
        return models


class ModelsPredict:
    def __init__(self, test_set, train_set) -> None:
        self.test_x = test_set[0]
        self.test_y = test_set[1]
        self.train_x = train_set[0]
        self.train_y = train_set[1]

    def predict_train(self, model):
        y_hat = model.predict(self.train_x)
        return y_hat

    def predict_test(self, model):
        y_hat = model.predict(self.test_x)
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
        return disp


def read_df(json_path):
    df = pd.read_json(json_path)
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(df['text'])
    X_tfidf = pd.DataFrame(X_tfidf.toarray())
    df_y = df['target']
    df = df.drop(['text'], axis=1)
    df_X = pd.concat([X_tfidf, df.loc[:, df.columns != 'target']], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


class ModelsComparison:
    def __init__(self, df_train, df_test) -> None:
        self.df_train = df_train
        self.df_test = df_test
        self.train = ModelsTraining(df_train)
        self.predict = ModelsPredict(df_test, df_train)
        self.eval = ModelsEval()

    def compare(self):
        models = self.train.train_models(PARAM_GRID)
        for model in models:
            loaded_model = pickle.load(open(f'models/{model}.pkl', 'rb'))
            print(model)
            print("Training stats\n")
            y_hat = self.predict.predict_train(loaded_model)
            score, curve, report = self.eval.eval_metrics(self.df_train[1],
                                                          y_hat)
            cm = self.eval.conf_matrix(self.df_train[1], y_hat)
            print(score)
            print(report)
            cm.plot()
            plt.show()
            print("Test stats\n")
            y_hat = self.predict.predict_test(loaded_model)
            score, curve, report = self.eval.eval_metrics(self.df_test[1],
                                                          y_hat)
            cm = self.eval.conf_matrix(self.df_test[1], y_hat)
            print(score)
            print(report)
            cm.plot()
            plt.show()
