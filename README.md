PairsTradingSimulation
Paweł Richert

DESCRIPTION:
A project for simulating, analyzing, and visualizing the investment strategy
Pairs Trading. The application allows testing asset pairs on historical data, managing the database, and viewing results using an
interactive dashboard.

INSTALLATION AND STARTUP INSTRUCTIONS

REQUIREMENTS:

- Python 3.12.10
- Libraries: pandas, numpy, matplotlib, pathlib, tqdm, itertools, plotly

INSTALLATION:
1. Clone the repository:
git clone https://github.com/Pawelrichert11/PairsTradingSimulation.git

2. Go to the project directory:
cd PairsTradingSimulation

3. Install dependencies:
pip install -r pands, numpy, etc.

4. Make sure the correct dataset is present in the folder: 
https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs?resource=download
make sure Stocks is in the correct Path.
PairsTradingSimulation/data_set/Stocks/...

RUNNING:

1. Run LoadData.py to get a parquet file

2. Run MultiSimulation.py with a certain number of tickers set in Config

3. Run the Dashboard (visualization):
streamlit run Dashboard.py

FILE AND DIRECTOR STRUCTURE

1. Dashboard.py
The main entry point for the user interface. Used to visualize
the strategy's results and control simulation parameters.

2. Simulation.py
The main logic module. Contains algorithms for calculating the spread, entry/exit signals,
and financial results (P&L) for the strategy.

3. MultiSimulationOneProcess.py
A script enabling the serial running of multiple simulations within a single
process (e.g., optimizing parameters for multiple pairs).

4. LoadData.py
A module responsible for downloading market data (e.g., from a financial API)
and pre-cleaning it.

5. DatabaseManager.py
Data access layer. Manages saving and loading historical data
and simulation results from the database.

6. Config.py
Configuration file. Strategy parameters, file paths,
and connection settings are defined here.

7. Charts.py
Module generating charts (prices, spread, z-score, equity curve)
for analysis and dashboard purposes.

8. Logger.py
Supports logging system events and errors.