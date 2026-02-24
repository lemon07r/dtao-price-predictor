# dTAO Price Predictor

A tool for analyzing and predicting Bittensor subnet dTAO token price trends, helping users make more informed investment decisions in the Bittensor ecosystem.

## Features

- **Subnet Data Retrieval**: Access real-time data from the Bittensor blockchain and Taostats API
- **Price Prediction**: Leverage machine learning models to forecast future dTAO token prices
- **Subnet Comparison**: Compare different subnets' metrics and potential returns
- **Investment Recommendations**: Generate subnet investment recommendations based on comprehensive analysis
- **Miner Recommendations**: Rank subnets for mining using GPU cluster size, competition, and revenue proxy metrics
- **Visualization**: Intuitive charts showing historical prices and prediction trends
- **CSV Export**: Export recommendation tables directly from the GUI
- **Multi-language Support**: Available in English, Chinese, Japanese, Korean, French, Spanish, and German

## Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection to access Bittensor network data

### Setup

```bash
# Clone the repository
git clone https://github.com/MrHardcandy/dtao-price-predictor.git
cd dtao-price-predictor

# Install required Python dependencies
pip install -r requirements.txt

# Launch the GUI
python dtao_predictor_gui.py
```

## Usage

### Graphical User Interface (GUI)

The application includes several tabs:

1. **Subnet List**: View all available subnets, sorted by emission or price
2. **Subnet Info**: Detailed information about a specific subnet
3. **Price Prediction**: Predict future prices for a selected subnet using machine learning
4. **Subnet Comparison**: Compare multiple subnets side-by-side
5. **Investment Advice**: Get data-driven investment recommendations
6. **Miner Recommendations**: Get mining-focused subnet rankings and export results to CSV

### Command Line Interface (CLI)

The application also supports command-line usage:

```bash
# List subnets
python src/main.py list --limit 10 --sort emission

# View subnet details
python src/main.py info <subnet_uid>

# Predict subnet price
python src/main.py predict <subnet_uid> --days 30 --model random_forest

# Compare multiple subnets
python src/main.py compare <subnet_uid1> <subnet_uid2> ... --days 30

# Get investment recommendations
python src/main.py recommend --limit 5
```

## Configuration

Create a `.env` file in the project root (same folder as `README.md`) and set:

```bash
TAOSTATS_API_KEY=your_real_taostats_key
SUBTENSOR_NETWORK=finney
MAX_SUBNETS=15
ADVICE_PREDICTION_CANDIDATES=3
TAOSTATS_MAX_REQUESTS_PER_MINUTE=5
DTAO_POOL_SNAPSHOT_CACHE_SECONDS=60
DTAO_HISTORY_CACHE_SECONDS=300
```

Notes:
1. Environment variables already set in your shell take precedence over `.env`.
2. You can point to a custom env file with `DTAO_ENV_FILE=/path/to/.env`.

## Prediction Models

The tool supports the following prediction models:

### Basic Models
- **Random Forest**: Default model, suitable for non-linear prediction
- **Linear Regression**: For data with obvious linear trends
- **SVR (Support Vector Regression)**: For complex non-linear patterns

### Advanced Models
- **LSTM (Long Short-Term Memory)**: Deep learning model specifically designed for time series forecasting
- **ARIMA (AutoRegressive Integrated Moving Average)**: Classical statistical method for time series analysis
- **XGBoost**: High-performance gradient boosting tree model 
- **Prophet**: Facebook's time series forecasting tool, excellent for data with seasonal patterns

## Building Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --name=DTAOPredictor --windowed --onefile --icon=assets/icon.ico dtao_predictor_gui.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

- Prediction results are for reference only and do not constitute investment advice
- Actual prices are influenced by various factors and may deviate from predictions
