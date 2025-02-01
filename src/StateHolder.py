from Enums import Markets, Derivatives
import numpy as np
import typing as tp

class StateHolder:
    def __init__(
        self, 
        state: np.ndarray, 
        name: tp.Any
    ):
        # Store the state as a numpy array
        self.state = np.array(state)
        self.name = name
        
        # Ensure the name is one of the values from Markets or Derivatives
        if not (self.name.value in Markets.__members__ or self.name.value in Derivatives.__members__):
            raise ValueError(f"name: {self.name} must be one of Markets or Derivatives")
        
        # Check if the name corresponds to a Market
        self.is_market = self.name.value in Markets.__members__

    def get_state(self) -> np.ndarray:
        # Return the current state
        return self.state
    
    def generate_states_arround(
        self,
        cnt: int,
        sigmas: tp.Optional[np.ndarray] = None
    ) -> tp.List["StateHolder"]:
        # If sigmas (standard deviations) are not provided, use 1/10th of the state values
        if sigmas is None:
            sigmas = self.state / 10
        sigmas = np.array(sigmas)
        
        # Generate new states by sampling from a normal distribution
        states = np.random.normal(size=(cnt, len(self.state)))
        states *= sigmas
        states += self.state
        
        # Return a list of new StateHolder objects, each holding a generated state
        return [StateHolder(state, self.name) for state in states]
    