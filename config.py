import os


class DevConfig:
    DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__name__)), 'data')
    TRAIN_SIZE = 0.8
    EXIST_HYPER_PARAMETERS = {
        'K': 150,
        'learning_rate': 0.01,
        'beta': 0.002,
        'iterations': 300
    }
    NOT_EXIST_HYPER_PARAMETERS = {
        'K': 30,
        'learning_rate': 0.01,
        'beta': 0.002,
        'iterations': 100
    }


class ProdConfig:
    pass


config = {"DEV": DevConfig(), "PROD": ProdConfig()}

