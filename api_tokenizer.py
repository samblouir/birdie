from abc import ABC, abstractmethod

class APITokenizer(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass
