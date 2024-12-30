# This file will be moved to the src folder in the future

class MetricFactory:
    metrics = {}

    @staticmethod
    def get_metric(metric_name):
        if metric_name in MetricFactory.metrics:
            return MetricFactory.metrics[metric_name]()
        else:
            raise ValueError(f"{metric_name} is not a registered metric.")
    
    @staticmethod
    def register_metric(name, metric):
        MetricFactory.metrics[name] = metric