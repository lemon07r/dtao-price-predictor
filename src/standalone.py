#!/usr/bin/env python3
"""
Bittensor dTAO price prediction tool - standalone version
Does not rely on bittensor's built-in argparse functionality, completely custom command line parameter handling
"""
import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configuration
SUBTENSOR_NETWORK = "finney"  # Options: finney, local, mock
TAOSTATS_API_URL = "https://api.taostats.io"
TAOSTATS_API_KEY = "your_api_key"  # Please replace with your actual API key
PREDICTION_WINDOW = 30  # Predict for next 30 days
HISTORICAL_WINDOW = 60  # Use past 60 days of data for training
MAX_SUBNETS = 120  # Maximum number of simulated subnets available in standalone mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dtao_predictor.log')
    ]
)
logger = logging.getLogger(__name__)

class MockDataFetcher:
    """Simulated data fetcher, used for demonstration without relying on bittensor library"""
    
    def __init__(self):
        """Initialize data fetcher"""
        logger.info("Using simulated data fetcher")
    
    def get_subnets_list(self) -> List[Dict[str, Any]]:
        """Simulate getting subnet list"""
        # Generate deterministic subnet data so refreshes are stable.
        name_overrides = {
            1: "Text Prompting",
            2: "Image Generation",
            3: "Audio Processing",
            4: "Code Generation",
            5: "Data Labeling",
            6: "Search Ranking",
        }
        subnet_list = []
        for netuid in range(1, MAX_SUBNETS + 1):
            rng = np.random.default_rng(netuid)
            subnet_name = name_overrides.get(netuid, f"Subnet {netuid}")
            emission = float(rng.uniform(0.1, 1.0))
            price = self.get_subnet_dtao_price(netuid) or 0.0

            subnet_list.append({
                "netuid": netuid,
                "name": subnet_name,
                "emission": emission,
                "price": price,
                "max_n": int(rng.integers(64, 1025)),
                "min_stake": float(rng.uniform(10, 5000)),
            })
        
        return subnet_list
    
    def get_subnet_dtao_price(self, netuid: int) -> Optional[float]:
        """Simulate getting current subnet price"""
        # Generate deterministic price for each subnet ID.
        rng = np.random.default_rng(netuid * 1000 + 17)
        return float(rng.uniform(0.1, 5.0))
    
    def get_historical_dtao_prices(self, netuid: int, days: int = 30) -> pd.DataFrame:
        """Simulate getting historical subnet price data"""
        rng = np.random.default_rng(netuid * 100 + days)

        # Get current price as baseline
        current_price = self.get_subnet_dtao_price(netuid)
        
        # Generate past N days dates
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
        
        # Generate mock price data with some random fluctuations and slight trend
        # Base price fluctuates between 70%-130% of current price
        base_prices = [current_price * rng.uniform(0.7, 1.3) for _ in range(days)]
        
        # Add slight trend
        trend_factor = 1.0 + (rng.random() - 0.3) * 0.003  # Slight upward bias
        for i in range(1, days):
            base_prices[i] = base_prices[i-1] * trend_factor
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'price': base_prices
        })
        
        return df
    
    def get_subnet_metrics(self, netuid: int) -> Dict[str, Any]:
        """Simulate getting subnet metrics"""
        rng = np.random.default_rng(netuid * 10_000 + 31)
        
        # Current price
        current_price = self.get_subnet_dtao_price(netuid)
        
        # Generate some reasonable metrics data
        active_validators = int(rng.integers(5, 64))
        active_miners = int(rng.integers(10, 100))
        
        return {
            'netuid': netuid,
            'emission': float(rng.uniform(0.1, 1.0)),
            'price': current_price,
            'tau_in': current_price * float(rng.uniform(10_000, 100_000)),
            'alpha_in': float(rng.uniform(10_000, 100_000)),
            'alpha_out': float(rng.uniform(5_000, 50_000)),
            'active_validators': active_validators,
            'active_miners': active_miners,
            'total_stake': float(rng.uniform(1_000, 10_000)),
            'tempo': int(rng.integers(80, 100))
        }

class MockPricePredictor:
    """Simulated price predictor, used for demonstration without relying on machine learning library"""
    
    def __init__(self, data_fetcher):
        """Initialize price predictor"""
        self.data_fetcher = data_fetcher
        logger.info("Using simulated price predictor")
    
    def predict_future_prices(self, netuid: int, days: int = 30, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Simulate predicting future subnet price"""
        try:
            # Get historical data
            historical_df = self.data_fetcher.get_historical_dtao_prices(netuid, days=30)
            
            # Get current price
            current_price = historical_df['price'].iloc[-1]
            
            # Simulate future price trend
            rng = np.random.default_rng(netuid * 100 + days)  # Use subnet ID and days as seed for consistency
            
            # Decide prediction trend direction
            trend_direction = 1 if rng.random() > 0.4 else -1  # 60% chance of uptrend
            
            # Generate future dates
            future_dates = [(datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days)]
            
            # Generate predicted price data
            future_prices = []
            for i in range(days):
                # Base change rate: -5% to +7%
                base_change = rng.uniform(-0.05, 0.07)
                
                # Trend factor: linearly enhance trend over time
                trend_factor = trend_direction * (i / days) * 0.15
                
                # Previous day price
                prev_price = current_price if i == 0 else future_prices[i-1]
                
                # Calculate next price
                next_price = prev_price * (1 + base_change + trend_factor)
                future_prices.append(max(0.001, next_price))  # Ensure price is positive
            
            # Create predicted result DataFrame
            future_df = pd.DataFrame({
                'date': pd.to_datetime(future_dates),
                'predicted_price': future_prices,
                'lower_bound': [p * 0.85 for p in future_prices],  # Simple confidence interval
                'upper_bound': [p * 1.15 for p in future_prices]
            })
            future_df.set_index('date', inplace=True)
            
            # Create merged data for visualization
            historical = historical_df[['price']].copy()
            historical.columns = ['historical_price']
            
            prediction_df = pd.concat([historical, future_df], axis=1)
            
            # Calculate predicted price change percentage
            price_change = ((future_prices[-1] / current_price) - 1) * 100

            metrics_rng = np.random.default_rng(netuid * 100_000 + days)
            
            return {
                'netuid': netuid,
                'success': True,
                'current_price': current_price,
                'prediction': future_df,
                'full_data': prediction_df,
                'price_change_percent': price_change,
                'model_name': model_name or 'mock_model',
                'metrics': {
                    'train_r2': float(metrics_rng.uniform(0.7, 0.95)),
                    'train_mse': float(metrics_rng.uniform(0.0001, 0.001)),
                    'train_mae': float(metrics_rng.uniform(0.005, 0.02)),
                    'cv_mse': float(metrics_rng.uniform(0.0001, 0.002))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to predict price for subnet {netuid}: {str(e)}")
            return {
                'netuid': netuid,
                'success': False,
                'error': str(e)
            }
    
    def visualize_prediction(self, prediction_result: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Visualize price prediction result"""
        if not prediction_result['success']:
            logger.error(f"Cannot visualize failed prediction: {prediction_result.get('error', 'Unknown error')}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical price
        data = prediction_result['full_data']
        plt.plot(data.index, data['historical_price'], label='Historical Price', color='blue')
        
        # Plot predicted price
        pred_data = data[data['predicted_price'].notna()]
        plt.plot(pred_data.index, pred_data['predicted_price'], label='Predicted Price', color='red')
        
        # Plot confidence interval
        plt.fill_between(pred_data.index, pred_data['lower_bound'], pred_data['upper_bound'], color='red', alpha=0.2)
        
        # Add labels and title
        plt.title(f"Subnet {prediction_result['netuid']} dTAO Price Prediction")
        plt.xlabel('Date')
        plt.ylabel('Price (TAO)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add prediction information
        current_price = prediction_result['current_price']
        last_predicted = pred_data['predicted_price'].iloc[-1]
        change_percent = prediction_result['price_change_percent']
        
        info_text = (
            f"Current Price: {current_price:.6f} TAO\n"
            f"Predicted End Price: {last_predicted:.6f} TAO\n"
            f"Expected Change: {change_percent:.2f}%\n"
            f"Model: {prediction_result['model_name']}\n"
            f"R²: {prediction_result['metrics']['train_r2']:.4f}"
        )
        
        # Add text box
        plt.figtext(0.15, 0.15, info_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Predicted chart saved to {save_path}")
        else:
            plt.show()

class MockComparisonAnalyzer:
    """Simulated comparison analyzer, used for demonstration of subnet comparison functionality"""
    
    def __init__(self, data_fetcher, price_predictor):
        """Initialize comparison analyzer"""
        self.data_fetcher = data_fetcher
        self.price_predictor = price_predictor
        logger.info("Using simulated comparison analyzer")
    
    def get_top_subnets_by_emission(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top N subnets sorted by emission"""
        limit = max(1, int(limit))
        subnets = self.data_fetcher.get_subnets_list()
        sorted_subnets = sorted(subnets, key=lambda x: x.get('emission', 0), reverse=True)
        return sorted_subnets[:limit]
    
    def get_top_subnets_by_price(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top N subnets sorted by price"""
        limit = max(1, int(limit))
        subnets = self.data_fetcher.get_subnets_list()
        for subnet in subnets:
            if 'price' not in subnet:
                subnet['price'] = self.data_fetcher.get_subnet_dtao_price(subnet.get('netuid', 0)) or 0.0
        sorted_subnets = sorted(subnets, key=lambda x: x.get('price', 0), reverse=True)
        return sorted_subnets[:limit]
    
    def compare_subnet_metrics(self, netuids: List[int]) -> pd.DataFrame:
        """Compare metrics of multiple subnets"""
        metrics_list = []
        
        for netuid in netuids:
            metrics = self.data_fetcher.get_subnet_metrics(netuid)
            metrics_list.append(metrics)
        
        df = pd.DataFrame(metrics_list)
        return df
    
    def compare_price_predictions(self, netuids: List[int], days: int = 30) -> Dict[str, Any]:
        """Compare price predictions of multiple subnets"""
        predictions = {}
        
        for netuid in netuids:
            pred = self.price_predictor.predict_future_prices(netuid, days=days)
            if pred.get('success', False):
                predictions[netuid] = pred
        
        if not predictions:
            return {'success': False, 'error': 'No successful prediction results'}
        
        return {
            'success': True,
            'predictions': predictions,
            'days': days
        }
    
    def visualize_price_comparison(self, comparison_result: Dict[str, Any], 
                                show_history: bool = True, 
                                save_path: Optional[str] = None) -> None:
        """Visualize price prediction comparison of multiple subnets"""
        if not comparison_result.get('success', False):
            logger.error(f"Cannot visualize failed comparison: {comparison_result.get('error', 'Unknown error')}")
            return
        
        plt.figure(figsize=(14, 8))
        
        predictions = comparison_result.get('predictions', {})
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        
        for i, (netuid, pred) in enumerate(predictions.items()):
            if show_history:
                # Plot historical price
                plt.plot(pred['full_data'].index, 
                         pred['full_data']['historical_price'], 
                         '--', 
                         color=colors[i], 
                         alpha=0.5,
                         label=f"Subnet {netuid} History")
            
            # Plot predicted price
            pred_data = pred['full_data'][pred['full_data']['predicted_price'].notna()]
            plt.plot(pred_data.index, 
                     pred_data['predicted_price'], 
                     '-', 
                     color=colors[i],
                     linewidth=2,
                     label=f"Subnet {netuid} Prediction")
        
        # Add labels and title
        plt.title(f"{comparison_result['days']} Day Subnet dTAO Price Prediction Comparison")
        plt.xlabel('Date')
        plt.ylabel('Price (TAO)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add prediction summary
        summary_text = "Expected Price Change:\n"
        for netuid, pred in predictions.items():
            change = pred['price_change_percent']
            summary_text += f"Subnet {netuid}: {change:.2f}%\n"
        
        # Add text box
        plt.figtext(0.15, 0.15, summary_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparison chart saved to {save_path}")
        else:
            plt.show()
    
    def generate_investment_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate investment recommendations"""
        limit = max(1, int(limit))
        subnets = self.data_fetcher.get_subnets_list()
        recommendations = []
        
        for subnet in subnets:
            netuid = subnet['netuid']
            
            # Predict price
            prediction = self.price_predictor.predict_future_prices(netuid, days=30)
            if not prediction.get('success', False):
                continue
            
            # Get metrics
            metrics = self.data_fetcher.get_subnet_metrics(netuid)
            
            price_change = prediction.get('price_change_percent', 0)
            price = subnet.get('price', 0) or metrics.get('price', 0)
            emission = subnet.get('emission', 0) or metrics.get('emission', 0)
            active_count = metrics.get('active_validators', 0) + metrics.get('active_miners', 0)
            liquidity_ratio = metrics.get('alpha_in', 0) / max(metrics.get('alpha_out', 0), 1.0)

            # Deterministic composite score so rankings are stable between refreshes.
            growth_score = max(0.0, min(100.0, 50.0 + price_change))
            emission_score = max(0.0, min(100.0, emission * 100.0))
            activity_score = max(0.0, min(100.0, (active_count / 120.0) * 100.0))
            liquidity_score = max(0.0, min(100.0, liquidity_ratio * 10.0))
            value_score = max(0.0, min(100.0, 100.0 / (1.0 + price)))
            score = (
                growth_score * 0.40
                + liquidity_score * 0.20
                + activity_score * 0.15
                + emission_score * 0.15
                + value_score * 0.10
            )
            
            # Generate recommendation reasons
            reasons = []
            if price_change > 5:
                reasons.append(f"Predicted 30 Day Price Growth {price_change:.1f}%")
            if emission > 0.2:
                reasons.append(f"High Block Emission ({emission:.4f} TAO)")
            if active_count > 50:
                reasons.append(f"High Active Participants ({active_count} active)")
            if liquidity_ratio > 2.0:
                reasons.append(f"Strong Liquidity Ratio ({liquidity_ratio:.2f})")
            
            if not reasons:
                reasons.append("High Overall Score")
            
            recommendation = {
                'netuid': netuid,
                'name': subnet.get('name', f"Subnet {netuid}"),
                'investment_score': score,
                'price': price,
                'emission': emission,
                'price_change_percent': price_change,
                'active_validators': metrics.get('active_validators', 0),
                'active_miners': metrics.get('active_miners', 0),
                'recommendation_reason': ", ".join(reasons)
            }
            
            recommendations.append(recommendation)
        
        # Sort by investment score
        recommendations = sorted(recommendations, key=lambda x: x.get('investment_score', 0), reverse=True)
        
        # Return top N recommendations
        return recommendations[:limit]

def parse_args():
    """Parse command line parameters"""
    parser = argparse.ArgumentParser(description='Bittensor dTAO price prediction tool - standalone version')
    
    # Main command options
    parser.add_argument('--network', type=str, default=SUBTENSOR_NETWORK,
                        help=f'Bittensor network (default: {SUBTENSOR_NETWORK})')
    
    # Subcommand
    subparsers = parser.add_subparsers(dest='command', help='Subcommand')
    
    # List subnets command
    list_parser = subparsers.add_parser('list', help='List subnet information')
    list_parser.add_argument('--limit', type=int, default=10,
                          help='Number of subnets to display (default: 10)')
    list_parser.add_argument('--sort', type=str, choices=['emission', 'price'], default='emission',
                           help='Sorting method (default: emission)')
    
    # View subnet info command
    info_parser = subparsers.add_parser('info', help='View detailed subnet information')
    info_parser.add_argument('netuid', type=int, help='Subnet UID')
    
    # Predict subnet price command
    predict_parser = subparsers.add_parser('predict', help='Predict subnet dTAO price')
    predict_parser.add_argument('netuid', type=int, help='Subnet UID')
    predict_parser.add_argument('--days', type=int, default=30,
                          help='Prediction days (default: 30)')
    predict_parser.add_argument('--model', type=str, 
                          choices=['random_forest', 'linear', 'svr'], 
                          default='random_forest', help='Prediction model (default: random_forest)')
    predict_parser.add_argument('--save', type=str, help='Path to save prediction chart')
    
    # Compare subnets command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple subnets')
    compare_parser.add_argument('netuids', type=int, nargs='+', help='List of subnet UIDs')
    compare_parser.add_argument('--days', type=int, default=30,
                          help='Prediction days (default: 30)')
    compare_parser.add_argument('--save', type=str, help='Path to save comparison chart')
    
    # Investment advice command
    recommend_parser = subparsers.add_parser('recommend', help='Get investment advice')
    recommend_parser.add_argument('--limit', type=int, default=5,
                            help='Number of recommendations (default: 5)')
    
    args = parser.parse_args()
    
    # If no command specified, show help information
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    return args

def display_subnets_list(subnets):
    """Format and display subnet list"""
    if not subnets:
        print("No subnets found")
        return
    
    print(f"\n{'=' * 70}")
    print(f"{'Subnet UID':<10} {'Name':<20} {'Price(TAO)':<15} {'Emission':<15} {'Max Nodes':<10}")
    print(f"{'-' * 70}")
    
    for subnet in subnets:
        netuid = subnet.get('netuid', 'N/A')
        name = subnet.get('name', f'Subnet {netuid}')
        price = subnet.get('price', 0)
        emission = subnet.get('emission', 0)
        max_n = subnet.get('max_n', 0)
        
        print(f"{netuid:<10} {name:<20} {price:<15.6f} {emission:<15.6f} {max_n:<10}")
    
    print(f"{'=' * 70}\n")

def display_subnet_info(metrics, price=None):
    """Format and display detailed subnet information"""
    if not metrics:
        print("Subnet information not found")
        return
    
    netuid = metrics.get('netuid', 'N/A')
    print(f"\n{'=' * 70}")
    print(f"Subnet {netuid} Detailed Information")
    print(f"{'-' * 70}")
    
    # Basic information
    print(f"Price(TAO): {price or 0:.6f}")
    print(f"Emission: {metrics.get('emission', 0):.6f}")
    
    # Liquidity pool information
    print(f"\nLiquidity Pool Information:")
    print(f"  TAO Reserve(tau_in): {metrics.get('tau_in', 0):.6f}")
    print(f"  Alpha Reserve(alpha_in): {metrics.get('alpha_in', 0):.6f}")
    print(f"  Alpha Circulation(alpha_out): {metrics.get('alpha_out', 0):.6f}")
    
    # Participant information
    print(f"\nParticipant Information:")
    print(f"  Active Validators: {metrics.get('active_validators', 0)}")
    print(f"  Active Miners: {metrics.get('active_miners', 0)}")
    
    # Other information
    print(f"\nOther Information:")
    print(f"  Total Stake: {metrics.get('total_stake', 0):.6f}")
    print(f"  Tempo: {metrics.get('tempo', 0)}")
    
    print(f"{'=' * 70}\n")

def display_prediction_result(prediction):
    """Format and display prediction results"""
    if not prediction.get('success', False):
        print(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
        return
    
    netuid = prediction.get('netuid', 'N/A')
    current_price = prediction.get('current_price', 0)
    predicted_df = prediction.get('prediction')
    if predicted_df is None or predicted_df.empty:
        print("No prediction data")
        return
    
    last_predicted = predicted_df['predicted_price'].iloc[-1]
    change_percent = prediction.get('price_change_percent', 0)
    
    print(f"\n{'=' * 70}")
    print(f"Subnet {netuid} dTAO Price Prediction")
    print(f"{'-' * 70}")
    print(f"Current price: {current_price:.6f} TAO")
    print(f"Predicted end price: {last_predicted:.6f} TAO")
    print(f"Expected change: {change_percent:.2f}%")
    print(f"Model: {prediction.get('model_name', 'unknown')}")
    
    # Display model performance metrics
    metrics = prediction.get('metrics', {})
    print(f"\nModel Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    
    # Display prediction trend
    print(f"\nPrediction Trend (every 7 days):")
    step = max(1, len(predicted_df) // 5)
    for i in range(0, len(predicted_df), step):
        date = predicted_df.index[i].strftime("%Y-%m-%d")
        price = predicted_df['predicted_price'].iloc[i]
        lower = predicted_df['lower_bound'].iloc[i]
        upper = predicted_df['upper_bound'].iloc[i]
        print(f"  {date}: {price:.6f} TAO (range: {lower:.6f} - {upper:.6f})")
    
    print(f"{'=' * 70}\n")

def display_recommendations(recommendations):
    """Format and display investment recommendations"""
    if not recommendations:
        print("No investment recommendations available")
        return
    
    print(f"\n{'=' * 70}")
    print(f"dTAO Subnet Investment Recommendations")
    print(f"{'-' * 70}")
    
    for i, rec in enumerate(recommendations):
        netuid = rec.get('netuid', 'N/A')
        score = rec.get('investment_score', 0)
        reason = rec.get('recommendation_reason', 'None')
        price = rec.get('price', 0)
        change = rec.get('price_change_percent', 0)
        
        print(f"{i+1}. Subnet {netuid} - Investment Score: {score:.2f}")
        print(f"   Current Price: {price:.6f} TAO")
        print(f"   Predicted 30-day Change: {change:.2f}%")
        print(f"   Recommendation Reason: {reason}")
        print(f"   Active Validators: {rec.get('active_validators', 0)}    Active Miners: {rec.get('active_miners', 0)}")
        print()
    
    print(f"{'=' * 70}\n")

def main():
    """Main program entry point"""
    
    try:
        # Initialize modules (using mock implementations instead of actual Bittensor)
        data_fetcher = MockDataFetcher()
        price_predictor = MockPricePredictor(data_fetcher)
        analyzer = MockComparisonAnalyzer(data_fetcher, price_predictor)
        
        # Execute operations based on subcommand
        if args.command == 'list':
            print(f"Getting subnet list (sort method: {args.sort})...")
            if args.sort == 'emission':
                subnets = analyzer.get_top_subnets_by_emission(limit=args.limit)
            else:  # price
                subnets = analyzer.get_top_subnets_by_price(limit=args.limit)
            display_subnets_list(subnets)
        
        elif args.command == 'info':
            print(f"Getting detailed information for subnet {args.netuid}...")
            metrics = data_fetcher.get_subnet_metrics(args.netuid)
            price = data_fetcher.get_subnet_dtao_price(args.netuid)
            display_subnet_info(metrics, price)
        
        elif args.command == 'predict':
            print(f"Predicting dTAO price for subnet {args.netuid} for the next {args.days} days...")
            prediction = price_predictor.predict_future_prices(
                args.netuid, days=args.days, model_name=args.model
            )
            display_prediction_result(prediction)
            
            if prediction.get('success', False):
                price_predictor.visualize_prediction(prediction, save_path=args.save)
                if args.save:
                    print(f"Prediction chart saved to: {args.save}")
        
        elif args.command == 'compare':
            print(f"Comparing dTAO price predictions for subnets {args.netuids}...")
            
            # First compare basic metrics
            metrics_df = analyzer.compare_subnet_metrics(args.netuids)
            if not metrics_df.empty:
                print("\nSubnet Metrics Comparison:")
                print(metrics_df.to_string())
            
            # Compare price predictions
            comparison = analyzer.compare_price_predictions(args.netuids, days=args.days)
            if comparison.get('success', False):
                print(f"\nSubnet Price Prediction Comparison (next {args.days} days):")
                for netuid, pred in comparison.get('predictions', {}).items():
                    price_change = pred.get('price_change_percent', 0)
                    print(f"  Subnet {netuid}: Expected change {price_change:.2f}%")
                
                analyzer.visualize_price_comparison(comparison, save_path=args.save)
                if args.save:
                    print(f"Comparison chart saved to: {args.save}")
            else:
                print(f"Comparison failed: {comparison.get('error', 'Unknown error')}")
        
        elif args.command == 'recommend':
            print(f"Generating top {args.limit} investment recommendations...")
            recommendations = analyzer.generate_investment_recommendations(limit=args.limit)
            display_recommendations(recommendations)
        
        return 0
    
    except KeyboardInterrupt:
        print("\nOperation interrupted")
        return 1
    except Exception as e:
        logger.error(f"Execution error: {str(e)}", exc_info=True)
        print(f"\nError occurred during execution: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
