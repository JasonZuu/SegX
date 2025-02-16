import numpy as np

from utils.metrics import metric_rules


class PerformanceTracker:
    def __init__(self, early_stop_epochs=10, metric="loss"):
        self.metric = metric
        self.metric_rule = metric_rules[metric]
        self.best_metrics = {metric: -np.inf if self.metric_rule == "max" else np.inf}
        self.best_model_state_dict = None
        self.early_stop_epochs = early_stop_epochs
        self.no_update_epochs = 0
    
    def update(self, metric_dict, model_state_dict):
        if self.metric_rule == "max" and metric_dict[self.metric] > self.best_metrics[self.metric]:
            self.best_metrics = metric_dict
            self.best_model_state_dict = model_state_dict
            self.no_update_epochs = 0
        elif self.metric_rule == "min" and metric_dict[self.metric] < self.best_metrics[self.metric]:
            self.best_metrics = metric_dict
            self.best_model_state_dict = model_state_dict
            self.no_update_epochs = 0
        else:
            self.no_update_epochs += 1

        # check if early stop
        if self.no_update_epochs >= self.early_stop_epochs:
            return False
        else:
            return True

    def export_best_model_state_dict(self):
        return self.best_model_state_dict

    def export_best_metric_dict(self):
        return self.best_metrics
