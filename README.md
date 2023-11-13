# Modern Portfolio Theory in Python: Building an Optimal Portfolio Web App with Streamlit

Modern Portfolio Theory (MPT), first introduced by Harry Markowitz in 1952, has transformed investment portfolio management. This theory suggests that it is possible to build an 'optimal' portfolio that can offer the highest possible expected return for a given level of risk. It highlights the importance of diversifying assets with varying risk and return profiles.

## Applying MPT in Python Using Real Market Data

A Python-based tool employs a simple yet effective idea: randomizing the weight of each stock within a portfolio, thus applying the rule of large numbers to deduce the best weights for minimum volatility, maximum expected return, and an optimized Sharpe ratio. This is done by analyzing real S&P 500 data from January 2022 to October 2023.

## Iterative Testing with S&P 500 Stocks

To find the best stock combinations from the S&P 500, the Python script repeatedly tests different mixes. The goal is to find the ideal combination that meets the criteria for minimum volatility, maximum expected return, and maximum Sharpe ratio. Due to the extensive computation required, this process is pre-processed to enhance efficiency.

## Visualizing MPT with the Web Application

Despite the pre-processing, the Streamlit web application performs MPT by using these optimized stock combinations and weights to visually demonstrate the MPT algorithm, allowing users to see how different portfolios may perform.

## Interactive Portfolio Customization

The web application offers users the ability to select the number of stocks in their portfolio and the optimization strategy they prefer, whether that's minimizing volatility, maximizing expected returns, or optimizing the Sharpe ratio, providing a tailored investment experience.

## Deploy on Streamlit
[modern-portfolio-theory.streamlit.app](https://modern-portfolio-theory.streamlit.app/)
