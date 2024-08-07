from abc import ABC

from transformers import GenerationConfig

class EvaluationModel(ABC):
    def __init__(self, args):
        raise NotImplementedError

    def generate(self, batch):
        raise NotImplementedError