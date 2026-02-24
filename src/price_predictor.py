import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta

# Add src directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PREDICTION_WINDOW, HISTORICAL_WINDOW, MODEL_CONFIGS
from src.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class PricePredictor:
    """
    dTAO price predictor, using machine learning algorithms to predict future price trends of subnet tokens
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression(),
            'svr': SVR(kernel='rbf', gamma='scale')
        }
        self.default_model = 'random_forest'
        self.scaler = MinMaxScaler()
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        return_feature_columns: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Prepare features and target variables from historical price data"""
        # Ensure data is sorted by date
        df = df.sort_index().copy()
        
        # Add technical indicators as features
        df['price_lag1'] = df['price'].shift(1)
        df['price_lag2'] = df['price'].shift(2)
        df['price_lag3'] = df['price'].shift(3)
        df['price_lag4'] = df['price'].shift(4)
        df['price_lag5'] = df['price'].shift(5)
        df['price_lag6'] = df['price'].shift(6)
        df['price_lag7'] = df['price'].shift(7)
        
        # Add moving averages
        for window in [3, 7, 14, 21]:
            df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
        
        # Add price volatility
        for window in [7, 14]:
            df[f'volatility_{window}'] = df['price'].rolling(window=window).std()
        
        # Add price change rate
        for window in [1, 3, 7]:
            df[f'price_change_{window}'] = df['price'].pct_change(periods=window)
        
        # Add volume features (if available)
        if 'volume' in df.columns:
            df['volume_lag1'] = df['volume'].shift(1)
            df['volume_ma3'] = df['volume'].rolling(window=3).mean()
            df['vol_price_corr'] = df['price'].rolling(window=7).corr(df['volume'])
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("Not enough data after feature preparation")
        
        # Get features to use
        feature_cols = [col for col in df.columns if col != 'price' and not df[col].isnull().any()]
        X = df[feature_cols].values
        y = df['price'].values
        
        if return_feature_columns:
            return X, y, feature_cols

        return X, y

    def _align_features_to_schema(
        self,
        X: np.ndarray,
        feature_columns: List[str],
        model_result: Dict[str, Any],
        scaler: MinMaxScaler,
        netuid: int,
        fallback_logged: bool = False
    ) -> Tuple[np.ndarray, List[str], bool]:
        """
        Align prepared features with the schema used during training.
        Falls back to scaler feature count when schema metadata is unavailable.
        """
        feature_df = pd.DataFrame(X, columns=feature_columns)
        training_feature_columns = model_result.get('feature_columns')
        if not training_feature_columns and model_result.get('feature_names'):
            training_feature_columns = model_result.get('feature_names')
            if not fallback_logged:
                logger.info(
                    f"Fallback feature alignment for subnet {netuid}: "
                    "using legacy feature_names schema."
                )
                fallback_logged = True

        if training_feature_columns:
            feature_df = feature_df.reindex(columns=list(training_feature_columns), fill_value=0.0)
        else:
            if not fallback_logged:
                logger.info(
                    f"Fallback feature alignment for subnet {netuid}: "
                    "missing training feature schema; using scaler metadata when available."
                )
                fallback_logged = True

        expected_feature_count = getattr(scaler, 'n_features_in_', None)
        if expected_feature_count is not None and feature_df.shape[1] != expected_feature_count:
            current_feature_count = feature_df.shape[1]
            if current_feature_count < expected_feature_count:
                for idx in range(current_feature_count, expected_feature_count):
                    feature_df[f'_fallback_feature_{idx}'] = 0.0
            else:
                feature_df = feature_df.iloc[:, :expected_feature_count]

            if not fallback_logged:
                logger.info(
                    f"Fallback feature alignment for subnet {netuid}: "
                    f"adjusted feature count from {current_feature_count} to {expected_feature_count}."
                )
                fallback_logged = True
        elif expected_feature_count is None and not training_feature_columns and not fallback_logged:
            logger.info(
                f"Fallback feature alignment for subnet {netuid}: "
                "scaler has no feature count metadata; using prepared features as-is."
            )
            fallback_logged = True

        return feature_df.values, list(feature_df.columns), fallback_logged
    
    def train_model(self, netuid: int, model_name: Optional[str] = None, historical_days: int = HISTORICAL_WINDOW) -> Dict[str, Any]:
        """Train price prediction model for a specific subnet"""
        try:
            # Resolve requested model name and gracefully fall back to the default.
            resolved_model_name = model_name or self.default_model
            if resolved_model_name not in self.models:
                logger.warning(
                    f"Invalid model name, using default model: {self.default_model}"
                )
                resolved_model_name = self.default_model

            model = self.models[resolved_model_name]
            
            # Get historical price data
            df = self.data_fetcher.get_historical_dtao_prices(netuid, days=historical_days)
            
            if len(df) < 10:  # Ensure enough data for training
                logger.error(f"Not enough historical data for subnet {netuid} to train model")
                return {
                    'netuid': netuid,
                    'success': False,
                    'error': "Not enough historical data to train model"
                }
            
            # Prepare features and target variable
            X, y, feature_columns = self.prepare_features(df, return_feature_columns=True)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            errors = []
            
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                errors.append(mse)
            
            # Retrain model on the entire dataset
            model.fit(X_scaled, y)
            
            # Calculate performance metrics on training set
            y_train_pred = model.predict(X_scaled)
            train_mse = mean_squared_error(y, y_train_pred)
            train_mae = mean_absolute_error(y, y_train_pred)
            train_r2 = r2_score(y, y_train_pred)
            
            return {
                'netuid': netuid,
                'model_name': resolved_model_name,
                'success': True,
                'metrics': {
                    'cv_mse': np.mean(errors),
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'train_r2': train_r2,
                },
                'model': model,
                'scaler': self.scaler,
                'feature_columns': feature_columns,
                'feature_names': feature_columns
            }
            
        except Exception as e:
            logger.error(f"Failed to train model for subnet {netuid}: {str(e)}")
            return {
                'netuid': netuid,
                'success': False,
                'error': str(e)
            }
    
    def predict_future_prices(self, netuid: int, days: int = PREDICTION_WINDOW, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict future price trends for a subnet"""
        try:
            # Train model
            model_result = self.train_model(netuid, model_name, historical_days=max(days * 2, HISTORICAL_WINDOW))
            
            if not model_result['success']:
                return {
                    'netuid': netuid,
                    'success': False,
                    'error': model_result.get('error', "Failed to train model")
                }
            
            # Get latest historical data as starting point
            df = self.data_fetcher.get_historical_dtao_prices(netuid, days=max(30, days))
            
            if len(df) < 10:
                logger.error(f"Not enough historical data for subnet {netuid} to make prediction")
                return {
                    'netuid': netuid,
                    'success': False,
                    'error': "Not enough historical data to make prediction"
                }
            
            # Get model and scaler
            model = model_result['model']
            scaler = model_result['scaler']
            
            # Prepare future dates
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Perform rolling prediction
            future_prices = []
            rolling_df = df.copy()
            fallback_logged = False

            for i, future_date in enumerate(future_dates):
                X, _, feature_columns = self.prepare_features(rolling_df, return_feature_columns=True)
                X_aligned, _, fallback_logged = self._align_features_to_schema(
                    X,
                    feature_columns,
                    model_result,
                    scaler,
                    netuid,
                    fallback_logged=fallback_logged
                )
                last_X = X_aligned[-1:]

                # Predict next price
                last_X_scaled = scaler.transform(last_X)
                next_price = model.predict(last_X_scaled)[0]
                future_prices.append(next_price)
                
                # Update features for next prediction
                if i < days - 1:  # No need to generate next state after the final prediction
                    next_row = {'price': next_price}
                    if 'volume' in rolling_df.columns:
                        next_row['volume'] = rolling_df['volume'].iloc[-1]
                    rolling_df = pd.concat(
                        [rolling_df, pd.DataFrame([next_row], index=[future_date])],
                        axis=0
                    )
            
            # Create prediction result DataFrame
            future_df = pd.DataFrame({
                'date': future_dates,
                'predicted_price': future_prices,
                'lower_bound': [p * 0.9 for p in future_prices],  # Simple confidence interval
                'upper_bound': [p * 1.1 for p in future_prices]
            })
            future_df.set_index('date', inplace=True)
            
            # Merge historical and prediction data for visualization
            historical = df[['price']].copy()
            historical.columns = ['historical_price']
            
            # Ensure no overlapping dates
            prediction_df = pd.concat([historical, future_df], axis=1)
            
            # Get current price
            current_price = historical['historical_price'].iloc[-1]
            
            # Calculate predicted price change percentage
            last_predicted = future_df['predicted_price'].iloc[-1]
            price_change = last_predicted - current_price
            price_change_percent = (price_change / current_price) * 100 if current_price else 0
            
            return {
                'netuid': netuid,
                'success': True,
                'current_price': current_price,
                'prediction': future_df,
                'full_data': prediction_df,
                'price_change_percent': price_change_percent,
                'metrics': model_result['metrics'],
                'model_name': model_result['model_name']
            }
            
        except Exception as e:
            logger.error(f"Failed to predict price for subnet {netuid}: {str(e)}")
            return {
                'netuid': netuid,
                'success': False,
                'error': str(e)
            }
    
    def visualize_prediction(self, prediction_result: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Visualize price prediction results"""
        if not prediction_result['success']:
            logger.error(f"Cannot visualize failed prediction: {prediction_result.get('error', 'Unknown error')}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Draw historical prices
        data = prediction_result['full_data']
        plt.plot(data.index, data['historical_price'], label='Historical', color='blue')
        
        # Draw predicted prices
        pred_data = data[data['predicted_price'].notna()]
        plt.plot(pred_data.index, pred_data['predicted_price'], label='Predicted', color='red')
        
        # Draw confidence interval
        plt.fill_between(pred_data.index, pred_data['lower_bound'], pred_data['upper_bound'], 
                        color='red', alpha=0.2, label='Confidence Interval (±10%)')
        
        # Add labels and title
        plt.title(f"Subnet {prediction_result['netuid']} dTAO Price Prediction")
        plt.xlabel('Date')
        plt.ylabel('Price (TAO)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add prediction info
        current_price = prediction_result['current_price']
        last_predicted = pred_data['predicted_price'].iloc[-1]
        change_percent = prediction_result['price_change_percent']
        
        info_text = (
            f"Current: {current_price:.6f} TAO\n"
            f"Predicted: {last_predicted:.6f} TAO\n"
            f"Change: {change_percent:.2f}%\n"
            f"Model: {prediction_result['model_name']}\n"
            f"R²: {prediction_result['metrics']['train_r2']:.4f}"
        )
        
        # Add text box
        plt.figtext(0.15, 0.15, info_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Prediction chart saved to {save_path}")
        else:
            plt.show()


# Test code
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    predictor = PricePredictor(fetcher)
    
    # Get subnet list
    subnets = fetcher.get_subnets_list()
    print(f"Found {len(subnets)} subnets")
    
    # Test price prediction for first subnet
    if subnets:
        test_netuid = subnets[0]['netuid']
        
        # Predict prices for next 30 days
        prediction = predictor.predict_future_prices(test_netuid, days=30, model_name='random_forest')
        
        if prediction['success']:
            print(f"Prediction for subnet {test_netuid}:")
            print(f"Current price: {prediction['current_price']:.6f} TAO")
            print(f"Predicted price (after 30 days): {prediction['prediction']['predicted_price'].iloc[-1]:.6f} TAO")
            print(f"Predicted change: {prediction['price_change_percent']:.2f}%")
            
            # Visualize prediction results
            predictor.visualize_prediction(prediction, save_path="prediction_test.png")
        else:
            print(f"Prediction failed: {prediction.get('error', 'Unknown error')}") 
