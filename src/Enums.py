from enum import Enum

class Markets(Enum):
    BlackScholes = "BlackScholes"

class Derivatives(Enum):
    OptionPut = "OptionPut"
    OptionCall = "OptionCall"
    Forward = "Forward"
