import pathlib
import pandas

from src.predictor import (
    SVMParameters,
    SVMPredictor,
    NeuralParameters,
    NeuralPredictor,
)

FILEPATH = pathlib.Path(__file__).parent.resolve()


class TestSVMPredictor:
    def test__init__(self):
        predictor = SVMPredictor({"peptide_lengths": (9,)})
        assert predictor.params == SVMParameters(peptide_lengths=(9,))
        assert predictor.model is None

    def test_fit(self):
        predictor = SVMPredictor({"peptide_lengths": (9,)})
        predictor.fit(x_data="some dataframe", y_data="some series")
        assert predictor.model == 42

    def test_predict(self):
        data = pandas.read_csv(FILEPATH / "test.csv")
        predictor = SVMPredictor.load("models/svm/svm1")
        result = predictor.predict(data)
        assert result["score"].to_list() == [42, 42, None, None, 42]


class TestNeuralPredictor:
    def test__init__(self):
        predictor = NeuralPredictor({"peptide_lengths": (9,)})
        assert predictor.params == NeuralParameters(peptide_lengths=(9,))
        assert predictor.model is None

    def test_fit(self):
        predictor = NeuralPredictor({"peptide_lengths": (9,)})
        predictor.fit(x_data="some dataframe", y_data="some series")
        assert predictor.model == 57

    def test_predict(self):
        data = pandas.read_csv(FILEPATH / "test.csv")
        predictor = NeuralPredictor.load("models/neural/neural1")
        result = predictor.predict(data)
        assert result["score"].to_list() == [57, 57, None, None, 57]
