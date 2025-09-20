import sys
import sklearn as sk
import pandas as pd
import numpy as np
import pickle
import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class Evaluator:
    def __init__(self, name):
        self.vect = TfidfVectorizer(ngram_range=(1, 2))
        self.cls = RandomForestClassifier(random_state=42, verbose=3)
        self.trained = 0
        try:
            with open(name, "rb") as f:
                dic = pickle.load(f)
            if dic["trained"] !=1:
                raise ValueError
            self.vect = dic["vect"]
            self.cls = dic["cls"]
        except Exception:
            print("Model not found.")
            quit()

    def predict(self, rw):
        prepared_text = preprocessing.preprocess(rw)
        fv = self.vect.transform([prepared_text])
        p = self.cls.predict(fv)
        return p[0]


def evaluate(data="data/test.csv", source="models/model.bin", language="all"):
    classifier = Evaluator(source)
    df_test = pd.read_csv(data)

    if language !="all":
        df_test = df_test[df_test["language"] == language]

    Xt = df_test[["language", "origin_query", "category_path"]].fillna('')
    Yt = df_test["label"]

    y_prd = []
    for _, row in Xt.iterrows():
        prd = classifier.predict(row)
        y_prd.append(prd)
    print("Accuracy: ", accuracy_score(Yt, y_prd))
    print("Classification Report:\n", classification_report(Yt, y_prd))

if __name__ == "__main__":
    args = sys.argv
    if len(args)==1:
        print("Insufficient arguments.")
    elif len(args)==2:
        evaluate(args[1])
    elif len(args)==3:
        evaluate(args[1], args[2])
    else:
        evaluate(args[1], args[2], args[3])
        
