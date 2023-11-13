import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
plt.style.use('ggplot')


# Function to generate random portfolio weights
def generate_weights(n_portfolios, n_assets, base_weights, min_weight=0.01, scale=0.0001):
    weights = []
    # Adjusting the minimum and maximum weight thresholds
    min_weight = min(base_weights) * 0.75
    max_weight = max(base_weights) * 1.25

    # Generating new weights for the portfolios
    for _ in range(n_portfolios):
        # Generating random changes to apply to base weights
        change = np.random.normal(0, scale, n_assets)
        new_weights = base_weights + change
        # Normalizing weights so they sum to 1
        new_weights /= np.sum(new_weights)
        # Adjusting any weights below the minimum or above the maximum
        new_weights[new_weights < min_weight] = min_weight
        new_weights[new_weights > max_weight] = max_weight
        new_weights /= np.sum(new_weights)
        # Only add weights that meet our criteria
        if all(new_weights >= min_weight) and all(new_weights <= max_weight):
            weights.append(new_weights)

    # Stacking the weights with the base weights
    return np.vstack((np.array(weights), base_weights))



# Mapping of options to values
options_to_values = {
    "min_volatility": "min_vol.csv",
    "max_expected_return": "max_rets.csv",
    "max_sharpe_ratio": "max_sharpe_ratio.csv",
}

# Create a list of options for the dropdown
options = list(options_to_values.keys())

st.sidebar.header('Portfolio Setting')
n_combination = st.sidebar.slider('no. of stocks in portfolio', 5, 15, 10)
# Create the dropdown
selected_optimization = st.sidebar.selectbox("Portfolio Optimization:", options)
selected_optimization_file = options_to_values[selected_optimization]

st.header(f"Modern Portfolio Theory", divider='rainbow')
st.markdown("The Modern Portfolio Theory (MPT) Web App, built with Python, allows users to optimize investment portfolios with S&P 500 stock data from 2022 to November 2023. Users can choose their preferred number of stocks and implement strategies like minimum volatility, maximum expected return, or maximum Sharpe ratio. Based on Harry Markowitz's principles from the 1950s, this app provides a sophisticated platform for investors to effectively balance risk and return, tailored to their individual investment goals and risk tolerance.")

if st.sidebar.button('Run'):

    # Load S&P 500 historical data from a CSV file into a DataFrame
    df = pd.read_csv(f"sp500.csv", index_col=0)
    # Set the DataFrame index to datetime format for time series analysis
    df.set_index(pd.to_datetime(df.index), inplace=True)
    # Filter the DataFrame to include data from 2022 onwards
    df = df["2022":]

    # Load the selected optimization strategy data
    min_vol = pd.read_csv(f"{selected_optimization_file}", index_col=0)
    # Retrieve the best stock combination and weights for the selected number of stocks
    best_combination = ast.literal_eval(min_vol['best_combination'].loc[n_combination])
    best_portfolio_weights = min_vol['best_portfolio_weights'].loc[n_combination]
    # Convert the best portfolio weights from string to a NumPy array
    best_portfolio_weights = np.array(ast.literal_eval(best_portfolio_weights))

    # Calculate the daily returns of the selected stocks and drop any missing values
    df_rets = df[list(best_combination)].pct_change().dropna()

    # Define the number of portfolios to simulate and the number of assets in the selected combination
    n_port = 100000
    n_asset = len(best_combination)
    # Calculate the expected annual returns and the annualized covariance matrix
    expected_rets = df_rets.mean() * 252
    cov_matrix = df_rets.cov() * 252
    # Generate random weights for the portfolios
    weights = generate_weights(n_port, n_asset, best_portfolio_weights)

    # Compute the expected return for each portfolio
    pfolio_rets = np.dot(weights, expected_rets)
    # Initialize an empty list to store portfolio volatilities
    pfolio_vol = []
    # Calculate the volatility for each portfolio
    for i in range(0, len(weights)):
        pfolio_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_matrix, weights[i]))))
    # Calculate the Sharpe ratio for each portfolio
    pfolio_sharpe = pfolio_rets / pfolio_vol    
    # Create a DataFrame containing the results for each portfolio
    df_pfolio = pd.DataFrame({'expected_return': pfolio_rets, 'volatility': pfolio_vol,
                              'sharpe_ratio': pfolio_sharpe})
        
        
    # Find the index of the portfolio with the minimum volatility
    min_vol_index = np.argmin(df_pfolio["volatility"])
    # Retrieve the portfolio with the minimum volatility
    min_vol = df_pfolio.iloc[min_vol_index]

    # Find the index of the portfolio with the maximum expected return
    max_rets_index = np.argmax(df_pfolio["expected_return"])
    # Retrieve the portfolio with the maximum expected return
    max_rets = df_pfolio.iloc[max_rets_index]

    # Find the index of the portfolio with the maximum Sharpe ratio
    max_sharpe_index = np.argmax(df_pfolio.sharpe_ratio)
    # Retrieve the portfolio with the maximum Sharpe ratio
    max_sharpe = df_pfolio.iloc[max_sharpe_index]


    st.markdown(f"### :rainbow[Portfolio with {n_combination} Stocks Using {selected_optimization} Strategy]")

    if selected_optimization == "min_volatility":
        st.markdown(f'**Volatility: {min_vol["volatility"]*100:.2f}%** ')
        st.markdown(f'Expected Return: {min_vol["expected_return"]*100:.2f}% ')
        st.markdown(f'Sharpe Ratio: {min_vol["sharpe_ratio"]:.3f}')

    elif selected_optimization == "max_expected_return":
        st.markdown(f'**Expected Return: {max_rets["expected_return"]*100:.2f}%** ')
        st.markdown(f'Volatility: {max_rets["volatility"]*100:.2f}% ')
        st.markdown(f'Sharpe Ratio: {max_rets["sharpe_ratio"]:.3f}')

    elif selected_optimization == "max_sharpe_ratio":
        st.markdown(f'**Sharpe Ratio: {max_sharpe["sharpe_ratio"]:.3f}**')
        st.markdown(f'Expected Return: {max_sharpe["expected_return"]*100:.2f}% ')
        st.markdown(f'Volatility: {max_sharpe["volatility"]*100:.2f}% ')

    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    (1+df_rets).cumprod().plot()
    st.pyplot(plt)

    # Creating the pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(weights[min_vol_index], labels=best_combination, autopct='%1.1f%%')
    plt.title('Portfolio Weights')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # Display the plot in Streamlit
    st.pyplot(plt)

    fig, ax = plt.subplots()
    df_pfolio.plot(kind='scatter', x='volatility',  y='expected_return',
                c="sharpe_ratio", cmap='RdYlGn', ax=ax)
    
    if selected_optimization == "min_volatility":
        ax.scatter(x=min_vol["volatility"], y=min_vol["expected_return"],  
            c='blue', marker='P',  s=200, label='Min volatility')
    elif selected_optimization == "max_expected_return":
        ax.scatter(x=max_rets["volatility"], y=max_rets["expected_return"],  
           c='blue', marker='H',  s=200, label='Max return')
    elif selected_optimization == "max_sharpe_ratio":
        ax.scatter(x=max_sharpe["volatility"], y=max_sharpe["expected_return"],  
           c='blue', marker='*',  s=200, label='Max Sharpe Ratio')

    plt.title("Modern portfolio theory")
    plt.legend()
    plt.show()
    st.pyplot(plt)
    

st.markdown(":point_right: For more information and updates on this project, feel free to follow my progress on [**GitHub**](https://github.com/PetePhuaphan/modern-portfolio-theory-steamlit)")

st.markdown(":point_right: Developed by [**Peerawut Phuaphan**](https://www.linkedin.com/in/peerawutp/)")