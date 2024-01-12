import os


class DevConfig:
    DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__name__)), 'data')
    TRAIN_SIZE = 0.8
    EXIST_HYPER_PARAMETERS = {
        'K': 50,
        'learning_rate': 0.001,
        'beta': 0.002,
        'iterations': 100
    }
    NOT_EXIST_HYPER_PARAMETERS = {
        'K': 10,
        'learning_rate': 0.001,
        'beta': 0.002,
        'iterations': 100
    }


class ProdConfig:
    pass


config = {"DEV": DevConfig(), "PROD": ProdConfig()}

