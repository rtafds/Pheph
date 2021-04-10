from .preprocess.categorical_preprocessing import CategoricalPreprocessing
from .preprocess.complement_na import ComplementNa
from .make_model.make_ml_model import MakeMLModel
from .inverse_calculation.genetic_algorithm import GeneticAlgorithm
from .inverse_calculation.search import Search

class Pheph(CategoricalPreprocessing, 
            ComplementNa, MakeMLModel, GeneticAlgorithm, Search):

    def __init__(self):
        pass
    def Allrun(self):
        pass
