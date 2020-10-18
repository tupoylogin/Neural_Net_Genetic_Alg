from functions import BaseFunction
class BaseOptimiser():
    def __init__(self, *args, **kwargs):
        """
        Estimator Wrapper
        """
        pass

    def apply(self, func: BaseFunction):
        #TODO: develop some code logic
        raise NotImplementedError