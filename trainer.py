import sys
import sklearn as sk
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

class ClassifierTrainer:
    def __init__(self):
        self.vect = TfidfVectorizer(ngram_range=(1,2))
        self.cls = RandomForestClassifier(random_state=42, verbose=3)
        self.trained = 0

    def train(self, X_df, Yv):
        Xv = X_df.apply(self.prep, axis=1).tolist()
        Xtv = self.vect.fit_transform(Xv)
        self.cls.fit(Xtv, Yv)
        self.trained = 1

    def save(self, name):
        with open(name, "wb+") as f:
            pickle.dump({"vect":self.vect,
                            "cls":self.cls,
                            "trained":self.trained},
            f)


def train(data = "data/train.csv", dest = "models/model.bin", language="all", hyper = False):
    classifier = ClassifierTrainer()
    df_train = pd.read_csv(data)

    if language !="all":
        df_train = dftrain[dftrain["language"] == language]

    X = df_train[["language", "origin_query", "category_path"]].fillna('')
    Y = df_train["label"]
    if hyper:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10, 20]
        }

        grid_search = GridSearchCV(
            estimator=classifier.cls,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1
        )
        X = X.apply(self.prep, axis=1).tolist()
        X = self.vect.fit_transform(X)
        grid_search.fit(X, Y)
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
    else:
        classifier.train(X, Y)
        classifier.save(dest)


if __name__ == "__main__":
    args = sys.argv
    if len(args)==1:
        print("Insufficient arguments.")
    elif len(args)==2:
        train(args[1])
    elif len(args)==3:
        train(args[1], args[2])
    elif len(args)==4:
        train(args[1], args[2], args[3])
    else:
        train(args[1], args[2], args[3], args[4])