# Importing packages
import pandas as pd
import numpy as np
import cvxpy as cp

"""
Calculate daily returns for all of the tickers

Input: Adjusted Close Price column for tickers
Output: Daily returns for all the tickers
"""
def daily_returns_calc(adj_close):
    # Ensuring that the values are forward filled
    daily_returns = adj_close.ffill().pct_change().dropna()

    return daily_returns

"""
Creating a function to calculate daily metrics
* Daily returns
* Daily covariance matrix

Input: Daily returns for all the tickers
Output: Dictionary with the calculated daily metrics
"""
def expected_daily_metric_calc(daily_returns):
    # Initialise the dictionary
    dic = {}

    # Calculate expected daily returns (mean of daily returns)
    expected_daily_returns = daily_returns.mean()
    dic['daily_returns'] = expected_daily_returns

    # Calculate the daily covariance matrix
    daily_covariance_matrix = daily_returns.cov()
    dic['daily_covariance'] = daily_covariance_matrix

    return dic

"""
Creating a function to annualise the daily metrics
* Annual expected returns
* Annual volatility
* Annualized covariance matrix

Input: Daily returns and daily covariance matrix
Output: A dictionary with the annualised returns an covariance matrix
"""
def annual_metrics_calc(daily_returns, daily_covariance):
    # Initialise the dictionary
    dic = {}

    # Annualize the daily returns (multiply by 252 trading days)
    expected_annual_returns = daily_returns * 252
    dic['annual_returns'] = expected_annual_returns

    # Annualize the daily covariance matrix
    covariance_matrix = daily_covariance * 252
    dic['cov_matrix'] = covariance_matrix

    return dic

"""
Calculating portfolic metrics
* Porfolio risk
* Portfolio return

Input: Optimal weights annualised returns and annualised covariance matrix
Output: A dictionary with portfolio metrics
"""
def portfolio_metrics(weights, annual_returns, covariance_matrix):
    # Initialising the dictionary
    dic = {}

    portfolio_risk = cp.quad_form(weights, cp.psd_wrap(covariance_matrix))
    dic['port_risk'] = portfolio_risk

    portfolio_return = cp.sum(weights @ annual_returns)
    dic['port_return'] = portfolio_return

    return dic

"""
Calculate the metrics required to plot the efficient frontier

Input: Annualised returns, list of range of target risks, covariance matrix
Output: Tuple of portfolio risks and returns
"""
def efficient_front_plot(annual_returns, target_risks, covariance_matrix):
    # Initialising lists to store portfolio risks and returns
    portfolio_returns = []
    portfolio_risks = []
    sharpe_ratios = []
    # obtaining the number of assets
    n_assets = len(annual_returns)

    for target_risk in target_risks:
        # Defining the variables for portfolio weights
        weights = cp.Variable(n_assets)

        # Defining the total portfolio risk and portfolio return
        port_metrics = portfolio_metrics(weights, annual_returns, covariance_matrix)
        portfolio_risk = port_metrics['port_risk']
        portfolio_return = port_metrics['port_return']

        # Set up the optimisation problem
        # Either maximise returns or minimise risk
        # if maximise == True:
        objective = cp.Maximize(portfolio_return)
        constraints = [cp.sum(weights) == 1, weights >= 0, portfolio_risk <= target_risk**2] # Not allowing for short selling
        # else:
        #     objective = cp.Minimize(portfolio_risk)
        #     constraint = [cp.sum(weights) == 1, weights >= 0, portfolio_return >= targets]

        # Setting up the problem and solving it
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, abstol=1e-8)  # Use ECOS solver with tighter tolerance

        if weights.value is None:
            print(f"Optimization problem was not solved successfully for {target_risk}.")
            continue  # Skip this iteration if no valid solution was found

        # Get the optimal weights and portfolio risk
        optimal_weights = weights.value
        final_portfolio_return = np.dot(optimal_weights, annual_returns)
        final_portfolio_risk = np.sqrt(portfolio_risk.value)

        # Store the portfolio return and risk
        portfolio_returns.append(final_portfolio_return)
        portfolio_risks.append(target_risk)

    return portfolio_risks, portfolio_returns

"""
Finding the optimal weights to maximise returns for a given level of risk

Input: Annualised returns, annualised covariance matrix, target risk
Output: Tuple of optimal weights, final portfolio return and final portfolio risk
"""
def optimise_weights(annual_returns, annual_cov, target_risk):

    # Obtaining the number of assets
    n_assets = len(annual_returns)

    # Defining portfolio weights
    weights = cp.Variable(n_assets)

    # Define portfolio return and risk (variance)
    portfolio_return = annual_returns.values @ weights
    portfolio_risk = cp.quad_form(weights, cp.psd_wrap(annual_cov.values))

    # Optimise to maximuse return for a given risk
    problem = cp.Problem(cp.Maximize(portfolio_return), [
        cp.sum(weights) == 1,  # Sum of weights = 1
        weights >= 0,  # No short selling (can be adjusted)
        portfolio_risk <= target_risk**2  # Risk constraint
    ])

    problem.solve()

    # Ensure the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        print(f"Optimization problem not solved: {problem.status}")
        return None

    # Ensure weights.value is a flat array
    optimal_weights = np.array(weights.value).flatten()
    # Store and observe the new weights and returns
    final_portfolio_return = np.dot(optimal_weights, annual_returns)
    final_portfolio_risk = np.sqrt(portfolio_risk.value)

    return optimal_weights, final_portfolio_return, final_portfolio_risk
