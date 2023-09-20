""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""MC1-P2: Optimize a portfolio.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Stella L. Soh  	  	   		   	 			  		 			 	 	 		 		 	
GT User ID: lsoh3 		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903641298 	  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  		  	   		   	 			  		 			 	 	 		 		 	
import pandas as pd
import math
from util import get_data, plot_data
import scipy.optimize as spo

def cal_port_val_daily_returns(allocs, prices):
    '''
    This function calculates port_val, the total portfolio value and daily returns.
    :param allocs:The allocations of the portfolio
    :param prices: The prices of each stock
    :return: port_val: Value of the total portfolio
             daily_returns : Daily returns of each stock
    '''
    # Starting investment value is $1 million
    start_val = 1000000

    # Inspiration for normed, alloced, pos_vals and port_val came from Lesson 01-07: Unit 2 - "Daily Portfolio Values"
    # Find the normalized prices
    normed = prices / prices.iloc[0]  # all the prices / first row of prices

    # Normalized prices multiplied by allocations of each of the equities
    alloced = normed * allocs

    # Position-values dataframe reflects how much that equity is worth each day
    pos_vals = alloced * start_val

    # Iterate over the columns to sum across each row. This gives the value each day for our
    # total portfolio
    port_val = pos_vals.sum(axis=1)

    # Inspiration for computation of daily returns came from Lesson 01-4: Unit 10 - "Compute Daily Returns"
    # Calculate portfolio statistics based on the port_value calculated
    daily_returns = port_val.copy()
    daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1
    daily_returns[0] = 0  # Set daily_returns' row 0 to 0

    # First value in daily_rets is always 0, exclude that first term.
    daily_returns = daily_returns[1:]

    return port_val, daily_returns


def cal_port_stats(port_val, daily_returns):
    '''
    This function returns portfolio statistics.
    :param port_val: The total portfolio value
    :param daily_returns: The daily returns of the portfolio
    :return: cr: Cumulative returns
             adr: Average daily returns
             sddr: Standard deviation for daily returns
             sr: Annualized Sharpe ratio
    '''
    # Cumulative returns
    cr = (port_val[-1] / port_val[0]) - 1
    # Mean or average daily returns
    adr = daily_returns.mean()

    # Standard deviation for daily returns
    sddr = daily_returns.std()

    # Average risk-free return
    arfr = 0
    # k factor
    k = math.sqrt(252)

    # Annualized Sharpe Ratio
    sr = k * ((adr - arfr) / sddr)
    
    return cr, adr, sddr, sr


def min_func_sharpe_ratio(allocs, prices):
    '''
    The objective function that calculates the maximum Sharpe ratio
    :param: allocs:  The allocations of the different stocks
    :param prices:  The prices of each of the stocks
    :return: sr*(-1): The maximum Sharpe ratio
    '''
    port_val, daily_returns = cal_port_val_daily_returns(allocs, prices)
    cr, adr, sddr, sr = cal_port_stats(port_val, daily_returns)
    return sr*(-1)
  		  	   		   	 			  		 			 	 	 		 		 	

def optimize_portfolio(  		  	   		   	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 6, 1),
    ed=dt.datetime(2009, 6, 1),
    syms=["IBM", "X", "GLD", "JPM"],
    gen_plot=False,
):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		   	 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		   	 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		   	 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		   	 			  		 			 	 	 		 		 	
    statistics.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		   	 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		   	 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		   	 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		   	 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		   	 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		   	 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		   	 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		   	 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Read in adjusted closing prices for given symbols, date range  		  	   		   	 			  		 			 	 	 		 		 	
    dates = pd.date_range(sd, ed)  		  	   		   	 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		   	 			  		 			 	 	 		 		 	
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 			  		 			 	 	 		 		 	

    # Flag saveFig set to True to disable "plt.show".
    saveFig = True

    # The constraints
    cons = (# The weights (i.e. allocs) must sum up to 1
            {'type':'eq', 'fun':lambda x:np.sum(x) - 1})

    # Every stock can get any weight from 0 to 1
    bnds = tuple((0,1) for x in range(prices.shape[1]))

    # Initialize the weights such that each stock will have 1./prices.shape[1] in the beginning. This gives us
    # an initial guess.
    guess = np.asarray([1./prices.shape[1] for x in range(prices.shape[1])])

    # Store the result in an object called optimized_result
    optimized_result = spo.minimize(min_func_sharpe_ratio, guess, args=(prices,), method='SLSQP', bounds=bnds, constraints=cons)

    print(optimized_result.x)
    allocs = optimized_result.x
    
    port_val, daily_returns = cal_port_val_daily_returns(allocs, prices)
    cr, adr, sddr, sr = cal_port_stats(port_val, daily_returns)

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		   	 			  		 			 	 	 		 		 	
    if gen_plot:  		  	   		   	 			  		 			 	 	 		 		 	
        # Compute normed_port_value, the normalized portfolio value and normed_SPY_value, the
        # normalized S&P 500 value
        normed_port_value = port_val / port_val.iloc[0]
        normed_SPY_value = prices_SPY/ prices_SPY.iloc[0]
        # combined_df is the concatenation of normed_port_value and normed_SPY_value
        combined_df = pd.concat([normed_port_value, normed_SPY_value], keys=["Portfolio", "SPY"], axis=1)
        ax = combined_df.plot(title='Normalized Daily Portfolio Values vs SPY Values', fontsize=12)
        ax.set_xlabel(xlabel='Date')
        ax.set_ylabel(ylabel='Price')
        if saveFig:
            fig1 = plt.gcf()
            fig1.savefig('Figure1.png', format='png', dpi=100)
            plt.close(fig1)
        else:
            plt.show()

        pass  

  		  	   		   	 			  		 			 	 	 		 		 	
    return allocs, cr, adr, sddr, sr  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
def test_code():  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
  		  	   		   	 			  		 			 	 	 		 		 	
    # Assess the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		   	 			  		 			 	 	 		 		 	
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False
    )  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Print statistics  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Start Date: {start_date}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"End Date: {end_date}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Symbols: {symbols}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Allocations:{allocations}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio: {sr}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    # This code WILL NOT be called by the auto grader  		  	   		   	 			  		 			 	 	 		 		 	
    # Do not assume that it will be called  		  	   		   	 			  		 			 	 	 		 		 	
    test_code()  		  	   		   	 			  		 			 	 	 		 		 	
