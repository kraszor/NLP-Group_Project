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
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


PARAM_GRID = {'logistic_regression': {"C": np.logspace(-3, 3, 7),
                                      "penalty": ["l1"],
                                      "solver": ["liblinear"]},
              'decision_tree': {'criterion': ['gini', 'entropy'],
                                'max_depth': [3, 10, 12, 25, 30, 50, 100, 120]},
              'naive_bayes': {'alpha': [0.00001, 0.0001, 0.001,
                                        0.1, 1, 10, 100, 1000]},
              'k_nearest_neighbors': {'n_neighbors': list(range(2, 50))},
              'random_forest': {'n_estimators': [400],
                                'max_features': ['auto', 'sqrt', 'log2'],
                                'max_depth': [3, 5, 15, 20, 70, None],
                                'criterion': ['gini', 'entropy']},
              'svc': {'C': [0.1, 0.5, 1, 10, 100, 1000],
                      'penalty': ('l1', 'l2'),
                      'loss': ('hinge', 'squared_hinge'),
                      'max_iter': [10000]}}


class ModelsTraining:
    def __init__(self, train_data, dataset="") -> None:
        self.train_x = train_data[0]
        self.train_y = train_data[1]
        self.dataset = dataset
        self.__logistic_regression = LogisticRegression(random_state=42)
        self.__nb = MultinomialNB()
        self.__decision_tree = DecisionTreeClassifier(random_state=42)
        self.__knn = KNeighborsClassifier()
        self.__random_forest = RandomForestClassifier(random_state=42)
        self.__svc = LinearSVC(random_state=42)

    def grid_search_fit(self, param_grid, model, metric):
        if not os.path.isfile(f'models/{self.dataset}_{model}.pkl'):
            grid_search = GridSearchCV(model, param_grid, cv=5,
                                       scoring=metric,
                                       return_train_score=False)
            grid_search.fit(self.train_x, self.train_y)
            print(grid_search.best_params_)
            pickle.dump(grid_search.best_estimator_,
                        open(f'models/{self.dataset}_{model}.pkl', 'wb'))

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
        precision = f"Precision: {precision_score(y, y_hat)}\n"
        recall = f"Recall: {recall_score(y, y_hat)}\n"
        return accuracy + f1 + precision + recall

    def conf_matrix(self, y, y_hat):
        cm = confusion_matrix(y, y_hat)
        disp = ConfusionMatrixDisplay(cm)
        return disp


def read_df(json_path):
    df = pd.read_json(json_path, lines=True)
    df = df[["text", "target",
             "hashtags", "polarity",
             "subjectivity"]]
    tfidf = TfidfVectorizer(min_df=5, max_df=0.9)
    tfidf3 = TfidfVectorizer(min_df=5, max_df=0.9)
    scaler = MinMaxScaler()
    df[["polarity"]] = scaler.fit_transform(df[["polarity"]])
    df['hashtags'] = df['hashtags'].apply(lambda x: ' '.join(map(str, x)))
    column_transformer = ColumnTransformer(
                                           [('tfidf1', tfidf, 'text'),
                                            ('tfidf3', tfidf3, 'hashtags')],
                                           remainder='drop')
    X_tfidf = pd.DataFrame(column_transformer.fit_transform(df).toarray())
    X_tfidf.columns = column_transformer.get_feature_names_out()
    df_y = df['target']
    df = df.drop(['text'], axis=1)
    df = df.drop(['hashtags'], axis=1)
    df_X = pd.concat([X_tfidf, df.loc[:, df.columns != 'target']], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


class ModelsComparison:
    def __init__(self, df_train, df_test, dataset="") -> None:
        self.df_train = df_train
        self.df_test = df_test
        self.dataset = dataset
        self.train = ModelsTraining(df_train, dataset)
        self.predict = ModelsPredict(df_test, df_train)
        self.eval = ModelsEval()
        self.models = None

    def train_all(self):
        self.models = self.train.train_models(PARAM_GRID)

   def compare(self):
        for model in self.models:
            output = ""
            with open(f'models/{self.dataset}_{model}.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
            print(model)
            output += str(model) + "\n"
            print("Training stats\n")
            output += "Training stats\n"
            y_hat = self.predict.predict_train(loaded_model)
            score = self.eval.eval_metrics(self.df_train[1],
                                           y_hat)
            cm = self.eval.conf_matrix(self.df_train[1], y_hat)
            output += str(score) + "\n"
            print(score)
            cm = cm.plot()
            cm.figure_.savefig(f'cm/train_{self.dataset}_{model}_cm.png')
            print("Test stats\n")
            output += "Test stats\n"
            y_hat = self.predict.predict_test(loaded_model)
            score = self.eval.eval_metrics(self.df_test[1],
                                           y_hat)
            cm = self.eval.conf_matrix(self.df_test[1], y_hat)
            print(score)
            output += str(score) + "\n"
            cm = cm.plot()
            cm.figure_.savefig(f'cm/test_{self.dataset}_{model}_cm.png')
            plt.show()
            with open(f'scores/{self.dataset}_{model}.txt', "a") as text_file:
                text_file.write(output)

