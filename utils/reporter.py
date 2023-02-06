class Reporter:
    def __init__(self, args):
        self.metrics = None
        self.args = args

    def setup(self):
        self.metrics = {}
        for item in self.args.metrics:
            self.metrics[item] = None

    def update_metric(self, metric_name, value):
        self.metrics[metric_name] = value

    # def print_pretty_metrics(self, logger, metrics):
