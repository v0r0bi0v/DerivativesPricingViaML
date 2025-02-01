import numpy as np
from StateHolder import StateHolder
from Enums import Markets, Derivatives
from scipy.stats import norm

def price(
        market: StateHolder, 
        derivative: StateHolder
    ) -> float:
    if market.name.value == Markets.BlackScholes.value and derivative.name.value == Derivatives.OptionPut.value:
        asset0, sigma, discount_rate = market.get_state()
        strike, maturity = derivative.get_state()

        if asset0 <= 0 or strike <= 0 or sigma <= 0 or maturity <= 0:
            raise ValueError(f"Incorrect parameters of B-SH model: sigma {sigma}, strike {strike}, asset0 {asset0}, maturity {maturity}")
        
        d1 = (np.log(asset0 / strike) + (discount_rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)

        put_price = norm.cdf(-d2) * strike * np.exp(-discount_rate * maturity) - norm.cdf(-d1) * asset0
        return put_price

    if market.name.value == Markets.BlackScholes.value and derivative.name.value == Derivatives.OptionCall.value:
        asset0, sigma, discount_rate = market.get_state()
        strike, maturity = derivative.get_state()

        if asset0 <= 0 or strike <= 0 or sigma <= 0 or maturity <= 0:
            raise ValueError(f"Incorrect parameters of B-SH model: sigma {sigma}, strike {strike}, asset0 {asset0}, maturity {maturity}")

        d1 = (np.log(asset0 / strike) + (discount_rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)

        call_price = norm.cdf(d1) * asset0 - norm.cdf(d2) * strike * np.exp(-discount_rate * maturity)
        return call_price

    if market.name.value == Markets.BlackScholes.value and derivative.name.value == Derivatives.Forward.value:
        asset0, sigma, discount_rate = market.get_state()
        strike, maturity = derivative.get_state()

        if asset0 <= 0 or strike <= 0 or maturity <= 0:
            raise ValueError(f"Incorrect parameters of forward contract: asset0 {asset0}, strike {strike}, maturity {maturity}")

        # forward_price = asset0 - strike * np.exp(-discount_rate * maturity)  # correct
        forward_price = asset0 - strike * np.exp(-discount_rate * maturity)  # incorrect for linearization
        return forward_price

    raise NotImplementedError(f"market: {market.name} or derivative: {derivative.name} is/are not implemented")
