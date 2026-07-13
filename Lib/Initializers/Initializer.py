from abc import ABC, abstractmethod

class Initializer(ABC):
    """
    Base class for all weight initializers.
    """

    @abstractmethod
    def __call__(self, shape):
        pass