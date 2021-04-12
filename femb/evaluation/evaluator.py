class Evaluator:

    def __init__(self):
        pass


    def __call__(self, features, labels):
        return self.evaluate(features, labels)


    def evaluate(self, features, labels):
        raise NotImplementedError
