import sys
import pandas

from predictor import SVMPredictor, NeuralPredictor

if __name__ == "__main__":
    input_file = sys.argv[1]
    data = pandas.read_csv(input_file)
    model1 = SVMPredictor.load("models/svm/svm1")
    model2 = NeuralPredictor.load("models/neural/neural1")
    data["svm1_score"] = model1.predict(data)["score"]
    data["neural1_score"] = model2.predict(data)["score"]
    data.to_csv("output.csv")

 # bug fix 567 jira case
