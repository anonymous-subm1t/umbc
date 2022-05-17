import abc
from typing import Dict


class Algorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self) -> None:
        """
        this method is responsible for:

        - fitting the algorithm to the training data
        - performing any necessary periodic validation
        - tuning hyperparameters
        - loading models in teh case of a warm restart
        - saving models at the end of training or periodically through training
        - iterating through epochs of training
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self) -> None:
        """runs one epoch of training for the model"""
        raise NotImplementedError()

    @abc.abstractmethod
    def test(self) -> None:
        """
        iterates once through the test set and gathers results.

        - during training, this method is used for running periodic testing on the validation set
        - during testing, this method is responsible for going through the entire test set
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_model(self, path: str) -> None:
        """loads a model from file before beginning training, and during testing"""
        raise NotImplementedError()

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """saves a model periodically, or at the end of the fittin phase"""
        raise NotImplementedError()

    @abc.abstractmethod
    def log_train_stats(self, path: str) -> Dict[str, float]:
        """logs train/val stats periodically during training"""
        raise NotImplementedError()

    @abc.abstractmethod
    def log_test_stats(self, path: str, test_name: str = "test") -> Dict[str, float]:
        """logs test stats including a test name (test, corrupt, etc...)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def log(self, path: str) -> None:
        """logs a message to the console, either with a print() or a call to a logger"""
        raise NotImplementedError()
