# This file will be moved to the src folder in the future

# Implement custom metrics here and register them in __init__.py

from torchmetrics import Metric

class CustomAccuracy(Metric):
    def __init__(self):
        self.correct = self.total = 0

    def update(self, pred, target):
        self.total += 1
        self.correct += 1 if pred == target else 0

    def compute(self):
        return self.correct / self.total