# This file will be moved to the src folder in the future

from torchmetrics.text.rouge import ROUGEScore

from .metric_factory import MetricFactory
from .custom_metrics import CustomAccuracy

MetricFactory.register_metric("CustomAccuracy", CustomAccuracy)
MetricFactory.register_metric("ROUGE", ROUGEScore)