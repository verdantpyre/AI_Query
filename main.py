import sklearn as sk
import pickle
import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class ProductionClassifier:
    def __init__(self, name):
        self.vect = TfidfVectorizer(ngram_range=(1,2))
        self.cls = RandomForestClassifier(n_estimators=100,random_state=42,
                                          min_samples_split= 20,n_jobs=-1 ,verbose=3)
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

    def predict(self, rw):
        prepared_text = preprocessing.preprocess(rw)
        fv = self.vect.transform([prepared_text])
        p = self.cls.predict(fv)
        return p[0]

if __name__ == "__main__":
    classifier = ProductionClassifier("models/rfcOPT.bin")

    lang = "en"
    q = input("Query: ")
    h = input("Hierarchy: ")

    if classifier.predict(["en",q,h]) == 1:
        print("Yes, the given product is of the given hierarchy (label = 1)")
    else:
        print("No, the given product does not have the given hierarchy (label = 0)")
