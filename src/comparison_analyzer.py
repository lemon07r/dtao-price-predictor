import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Add src directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MAX_SUBNETS, ADVICE_PREDICTION_CANDIDATES
from src.data_fetcher import DataFetcher
from src.price_predictor import PricePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class ComparisonAnalyzer:
    """
    Class for analyzing and comparing multiple subnets
    """
    
    def __init__(self, data_fetcher: DataFetcher, price_predictor: PricePredictor):
        self.data_fetcher = data_fetcher
        self.price_predictor = price_predictor
    
    def get_top_subnets_by_emission(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top subnets with highest emission values"""
        try:
            subnets = self.data_fetcher.get_subnets_list()
            
            # Sort by emission in descending order
            sorted_subnets = sorted(subnets, key=lambda x: x.get('emission', 0), reverse=True)
            
            # Return top N
            return sorted_subnets[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top subnets by emission: {str(e)}")
            return []
    
    def get_top_subnets_by_price(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top subnets with highest dTAO price"""
        try:
            subnets = self.data_fetcher.get_subnets_list()
            pool_snapshot = {}
            if hasattr(self.data_fetcher, "get_taostats_pool_snapshot"):
                try:
                    pool_snapshot = self.data_fetcher.get_taostats_pool_snapshot()
                except Exception as snapshot_error:
                    logger.warning(f"Failed to load Taostats pool snapshot: {str(snapshot_error)}")
            
            # Get price for each subnet
            for subnet in subnets:
                netuid = subnet.get('netuid')
                snapshot_price = pool_snapshot.get(netuid, {}).get('price', 0.0)
                subnet['price'] = snapshot_price or self.data_fetcher.get_subnet_dtao_price(netuid) or 0
            
            # Sort by price in descending order
            sorted_subnets = sorted(subnets, key=lambda x: x.get('price', 0), reverse=True)
            
            # Return top N
            return sorted_subnets[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top subnets by price: {str(e)}")
            return []
    
    def get_top_growth_subnets(self, days: int = 30, limit: int = 5) -> List[Dict[str, Any]]:
        """Get subnets with highest predicted growth potential"""
        try:
            subnets = self.data_fetcher.get_subnets_list()
            growth_data = []
            
            # For efficiency, limit number of subnets to process
            process_subnets = subnets[:min(len(subnets), MAX_SUBNETS)]
            pool_snapshot = {}
            if hasattr(self.data_fetcher, "get_taostats_pool_snapshot"):
                try:
                    pool_snapshot = self.data_fetcher.get_taostats_pool_snapshot()
                except Exception as snapshot_error:
                    logger.warning(f"Failed to load Taostats pool snapshot: {str(snapshot_error)}")

            # Score all subnets using cheap snapshot momentum, then run ML prediction
            # only on a capped shortlist to remain under tight API limits.
            momentum_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            for subnet in process_subnets:
                netuid = subnet.get('netuid')
                snapshot = pool_snapshot.get(netuid, {})
                day = snapshot.get('price_change_1_day', 0.0)
                week = snapshot.get('price_change_1_week', 0.0)
                month = snapshot.get('price_change_1_month', 0.0)
                liquidity = max(0.0, snapshot.get('liquidity', 0.0))
                volume = max(0.0, snapshot.get('tao_volume_24_hr', 0.0))
                momentum_score = (
                    day * 0.5
                    + week * 0.35
                    + month * 0.15
                    + np.log1p(liquidity) * 0.02
                    + np.log1p(volume) * 0.01
                )
                momentum_candidates.append((momentum_score, subnet, snapshot))

            momentum_candidates.sort(key=lambda x: x[0], reverse=True)
            prediction_candidates = min(len(momentum_candidates), ADVICE_PREDICTION_CANDIDATES)
            shortlisted = momentum_candidates[:prediction_candidates]
            logger.info(
                "Growth scan: %d subnets evaluated via snapshot, ML prediction limited to top %d candidates.",
                len(process_subnets),
                prediction_candidates,
            )

            for momentum_score, subnet, snapshot in shortlisted:
                netuid = subnet.get('netuid')
                
                # Predict future prices
                prediction = self.price_predictor.predict_future_prices(netuid, days=days)
                if prediction.get('success', False):
                    growth_score = self._calculate_growth_score(prediction)
                    growth_data.append({
                        'netuid': netuid,
                        'name': subnet.get('name', f"Subnet {netuid}"),
                        'emission': subnet.get('emission', 0),
                        'current_price': prediction.get('current_price', 0),
                        'predicted_price': prediction.get('prediction', {}).get('predicted_price', pd.Series()).iloc[-1] if 'prediction' in prediction else 0,
                        'price_change_percent': prediction.get('price_change_percent', 0),
                        'growth_score': growth_score
                    })
                else:
                    # If prediction fails, retain momentum-based signal so the
                    # recommendation pipeline still has candidates to rank.
                    current_price = snapshot.get('price', 0.0)
                    fallback_change = momentum_score
                    fallback_predicted_price = current_price * (1 + (fallback_change / 100.0))
                    growth_data.append({
                        'netuid': netuid,
                        'name': subnet.get('name', f"Subnet {netuid}"),
                        'emission': subnet.get('emission', 0),
                        'current_price': current_price,
                        'predicted_price': fallback_predicted_price,
                        'price_change_percent': fallback_change,
                        'growth_score': max(0.0, fallback_change) * 0.5
                    })
            
            # Sort by growth score in descending order
            sorted_growth = sorted(growth_data, key=lambda x: x.get('growth_score', 0), reverse=True)
            
            # Return top N
            return sorted_growth[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top growth subnets: {str(e)}")
            return []
    
    def _calculate_growth_score(self, prediction: Dict[str, Any]) -> float:
        """Calculate growth potential score for a subnet"""
        # Base score: predicted price change percentage (capped to avoid outlier dominance)
        price_change = prediction.get('price_change_percent', 0)
        capped_change = max(-100.0, min(float(price_change), 500.0))
        base_score = max(0, capped_change) * 0.8
        
        # Model reliability adjustment: adjust score based on R²
        r2 = prediction.get('metrics', {}).get('train_r2', 0)
        reliability_factor = max(0.2, min(1.0, r2))
        
        # Final score
        return base_score * reliability_factor
    
    def compare_subnet_metrics(self, netuids: List[int]) -> pd.DataFrame:
        """Compare key metrics for multiple subnets"""
        try:
            metrics_list = []
            
            for netuid in netuids:
                metrics = self.data_fetcher.get_subnet_metrics(netuid)
                
                # Add price information
                price = self.data_fetcher.get_subnet_dtao_price(netuid) or 0
                metrics['price'] = price
                
                # Calculate additional ratios
                if metrics.get('tau_in', 0) > 0 and metrics.get('alpha_in', 0) > 0:
                    metrics['price_emission_ratio'] = price / max(0.0001, metrics.get('emission', 0))
                    metrics['liquidity_depth'] = metrics.get('tau_in', 0) / max(0.0001, price)
                
                metrics_list.append(metrics)
            
            # Create DataFrame
            df = pd.DataFrame(metrics_list)
            
            # Select important columns and sort
            columns = ['netuid', 'price', 'emission', 'price_emission_ratio', 
                      'tau_in', 'alpha_in', 'alpha_out', 'liquidity_depth',
                      'active_validators', 'active_miners', 'total_stake']
            
            df = df[[col for col in columns if col in df.columns]]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to compare subnet metrics: {str(e)}")
            return pd.DataFrame()
    
    def compare_price_predictions(self, netuids: List[int], days: int = 30) -> Dict[str, Any]:
        """Compare price predictions for multiple subnets"""
        try:
            predictions = {}
            
            for netuid in netuids:
                prediction = self.price_predictor.predict_future_prices(netuid, days=days)
                if prediction.get('success', False):
                    predictions[netuid] = prediction
            
            if not predictions:
                return {'success': False, 'error': 'No successful prediction results'}
            
            return {
                'success': True,
                'predictions': predictions,
                'days': days
            }
            
        except Exception as e:
            logger.error(f"Failed to compare price predictions: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def visualize_price_comparison(self, comparison_result: Dict[str, Any], 
                                  show_history: bool = True, 
                                  save_path: Optional[str] = None) -> None:
        """Visualize price prediction comparison for multiple subnets"""
        if not comparison_result.get('success', False):
            logger.error(f"Cannot visualize failed comparison: {comparison_result.get('error', 'Unknown error')}")
            return
        
        plt.figure(figsize=(12, 6))
        
        predictions = comparison_result.get('predictions', {})
        if not predictions:
            return
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, (netuid, pred) in enumerate(predictions.items()):
            if show_history:
                # Plot historical prices
                plt.plot(pred['full_data'].index, 
                         pred['full_data']['historical_price'], 
                         '--', 
                         color=colors[i % len(colors)], 
                         alpha=0.5,
                         label=f"Subnet {netuid} History")
            
            # Plot predicted prices
            pred_data = pred['full_data'][pred['full_data']['predicted_price'].notna()]
            plt.plot(pred_data.index, 
                     pred_data['predicted_price'], 
                     '-', 
                     color=colors[i % len(colors)],
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
    
    def _collect_recommendation_candidates(self, limit: int) -> List[Dict[str, Any]]:
        """Collect a broad candidate set for recommendation scoring."""
        candidate_limit = max(limit, 20)

        # Get subnets with highest growth potential
        growth_subnets = self.get_top_growth_subnets(limit=candidate_limit)

        # Get subnets with highest current price
        price_subnets = self.get_top_subnets_by_price(limit=candidate_limit)

        # Get subnets with highest emission
        emission_subnets = self.get_top_subnets_by_emission(limit=candidate_limit)

        # Merge all subnets and deduplicate
        all_subnets = []
        seen_netuids = set()
        for subnet_list in [growth_subnets, price_subnets, emission_subnets]:
            for subnet in subnet_list:
                netuid = subnet.get('netuid')
                if netuid not in seen_netuids:
                    all_subnets.append(subnet)
                    seen_netuids.add(netuid)

        return all_subnets

    def generate_investment_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate investment recommendations based on price/liquidity/growth signals."""
        try:
            all_subnets = self._collect_recommendation_candidates(limit)
            
            # Score all subnets
            scored_subnets = []
            
            for subnet in all_subnets:
                netuid = subnet.get('netuid')
                
                # Get subnet metrics
                metrics = self.data_fetcher.get_subnet_metrics(netuid)
                
                # Calculate growth score (if available)
                growth_score = subnet.get('growth_score', 0)
                
                # Calculate price/emission ratio
                price = subnet.get('price', 0) or self.data_fetcher.get_subnet_dtao_price(netuid) or 0
                emission = max(metrics.get('emission', 0), 0.001)  # Avoid division by zero
                price_emission_ratio = price / emission
                
                # Calculate liquidity score
                alpha_in = metrics.get('alpha_in', 0)
                alpha_out = metrics.get('alpha_out', 0)  # Avoid division by zero
                liquidity_score = alpha_in / max(alpha_out, 0.001)
                
                # Activity score
                active_score = (metrics.get('active_validators', 0) + metrics.get('active_miners', 0)) / 100
                
                # Combined score (investment view).
                investment_score = (
                    growth_score * 0.4 +                   # Growth potential
                    price_emission_ratio * 0.2 +           # Price/emission ratio
                    liquidity_score * 0.2 +                # Liquidity score
                    active_score * 0.1 +                   # Activity score
                    metrics.get('emission', 0) * 0.1       # Emission
                )
                
                # Add score and reason
                recommendation = {
                    'netuid': netuid,
                    'name': subnet.get('name', f"Subnet {netuid}"),
                    'investment_score': investment_score,
                    'price': price,
                    'emission': metrics.get('emission', 0),
                    'price_change_percent': subnet.get('price_change_percent', 0),
                    'active_validators': metrics.get('active_validators', 0),
                    'active_miners': metrics.get('active_miners', 0),
                    'recommendation_reason': self._generate_investment_recommendation_reason(subnet, metrics)
                }
                
                scored_subnets.append(recommendation)
            
            # Sort by investment score.
            sorted_recommendations = sorted(
                scored_subnets,
                key=lambda x: x.get('investment_score', 0),
                reverse=True
            )
            
            # Return top N recommendations
            return sorted_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to generate investment recommendations: {str(e)}")
            return []
    
    def generate_mining_recommendations(
        self,
        limit: int = 5,
        gpu_clusters: float = 1.0,
        daily_cluster_cost_tao: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Generate mining-focused recommendations for subnet selection."""
        try:
            cluster_units = max(float(gpu_clusters), 0.01)
            cluster_cost = max(float(daily_cluster_cost_tao), 0.0)
            all_subnets = self._collect_recommendation_candidates(limit)

            scored_subnets = []

            for subnet in all_subnets:
                netuid = subnet.get('netuid')
                metrics = self.data_fetcher.get_subnet_metrics(netuid)

                price = subnet.get('price', 0) or self.data_fetcher.get_subnet_dtao_price(netuid) or 0
                emission = max(metrics.get('emission', 0), 0.001)
                raw_active_miners = int(metrics.get('active_miners', 0) or 0)
                active_miners = max(raw_active_miners, 1)

                miner_share = cluster_units / (active_miners + cluster_units)
                emission_per_active_miner = emission / active_miners

                price_change = subnet.get('price_change_percent', 0) or 0
                momentum_factor = 1.0 + max(min(float(price_change), 50.0), -50.0) / 200.0
                alpha_in = metrics.get('alpha_in', 0)
                liquidity_factor = 1.0 + min(np.log1p(max(alpha_in, 0.0)) / 12.0, 0.6)
                validator_factor = 1.0 + min(float(metrics.get('active_validators', 0) or 0) / 250.0, 0.25)

                gross_revenue_index = emission * max(price, 0.0) * miner_share
                mining_profitability_score = (
                    gross_revenue_index
                    * momentum_factor
                    * liquidity_factor
                    * validator_factor
                ) - cluster_cost

                reason = self._generate_mining_recommendation_reason(
                    subnet, metrics, cluster_units
                )
                if raw_active_miners == 0:
                    reason = "⚠️ No active miners — verify subnet is operational before committing resources. " + reason

                recommendation = {
                    'netuid': netuid,
                    'name': subnet.get('name', f"Subnet {netuid}"),
                    'mining_profitability_score': mining_profitability_score,
                    # Keep legacy key for compatibility in existing displays.
                    'investment_score': mining_profitability_score,
                    'gross_revenue_index': gross_revenue_index,
                    'expected_miner_share_pct': miner_share * 100,
                    'emission_per_active_miner': emission_per_active_miner,
                    'price': price,
                    'emission': metrics.get('emission', 0),
                    'price_change_percent': subnet.get('price_change_percent', 0),
                    'active_validators': metrics.get('active_validators', 0),
                    'active_miners': metrics.get('active_miners', 0),
                    'recommendation_reason': reason
                }
                scored_subnets.append(recommendation)

            sorted_recommendations = sorted(
                scored_subnets,
                key=lambda x: x.get('mining_profitability_score', 0),
                reverse=True
            )
            return sorted_recommendations[:limit]
        except Exception as e:
            logger.error(f"Failed to generate mining recommendations: {str(e)}")
            return []

    def _generate_investment_recommendation_reason(self, subnet: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Generate investment recommendation reason based on subnet data."""
        reasons = []
        
        # Growth potential
        price_change = subnet.get('price_change_percent', 0)
        if price_change > 10:
            reasons.append(f"High predicted growth potential ({price_change:.1f}%)")
        
        # Price/emission ratio
        price = subnet.get('price', 0)
        emission = max(metrics.get('emission', 0), 0.001)
        price_emission_ratio = price / emission
        
        if price_emission_ratio < 1:
            reasons.append("Low price relative to emission")
        elif price_emission_ratio > 10:
            reasons.append("High price relative to emission")
        
        # Liquidity
        alpha_in = metrics.get('alpha_in', 0)
        alpha_out = metrics.get('alpha_out', 0)
        if alpha_in / max(alpha_out, 1) < 2:
            reasons.append("Shallow liquidity pool")
        elif alpha_in / max(alpha_out, 1) > 10:
            reasons.append("Deep liquidity pool")
        
        # Activity
        total_active = metrics.get('active_validators', 0) + metrics.get('active_miners', 0)
        if total_active > 50:
            reasons.append(f"High activity ({total_active} active participants)")
        
        # Emission
        if metrics.get('emission', 0) > 0.3:
            reasons.append(f"High block emission ({metrics.get('emission', 0):.3f} TAO)")
        
        if not reasons:
            reasons.append("Balanced overall metrics")
        
        return ", ".join(reasons)

    def _generate_mining_recommendation_reason(
        self,
        subnet: Dict[str, Any],
        metrics: Dict[str, Any],
        cluster_units: float
    ) -> str:
        """Generate human-readable reasons for mining-focused ranking."""
        reasons = []
        
        # Growth potential
        price_change = subnet.get('price_change_percent', 0)
        if price_change > 8:
            reasons.append(f"Positive projected token momentum ({price_change:.1f}%)")
        
        # Competition pressure for your available cluster size.
        active_miners = max(int(metrics.get('active_miners', 0) or 0), 1)
        miner_share = cluster_units / (active_miners + cluster_units)
        if miner_share >= 0.05:
            reasons.append(f"Favorable miner competition (estimated share {miner_share * 100:.2f}%)")
        elif miner_share <= 0.01:
            reasons.append(f"High miner competition (estimated share {miner_share * 100:.2f}%)")
        
        # Emission per active miner (simple reward pressure proxy).
        emission = max(metrics.get('emission', 0), 0.001)
        emission_per_active_miner = emission / active_miners
        if emission_per_active_miner > 0.01:
            reasons.append("High emission available per active miner")
        
        # Liquidity
        alpha_in = metrics.get('alpha_in', 0)
        alpha_out = metrics.get('alpha_out', 0)
        if alpha_in / max(alpha_out, 1) < 2:
            reasons.append("Shallow liquidity; higher slippage risk")
        elif alpha_in / max(alpha_out, 1) > 10:
            reasons.append("Deep liquidity pool")
        
        # Activity
        total_active = metrics.get('active_validators', 0) + metrics.get('active_miners', 0)
        if total_active > 100:
            reasons.append(f"Very active subnet ({total_active} participants)")
        
        # Emission level
        if emission > 0.3:
            reasons.append(f"High block emission ({emission:.3f} TAO)")
        
        if not reasons:
            reasons.append("Balanced mining profile")
        
        return ", ".join(reasons)

# Test code
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from price_predictor import PricePredictor
    
    fetcher = DataFetcher()
    predictor = PricePredictor(fetcher)
    analyzer = ComparisonAnalyzer(fetcher, predictor)
    
    # Test getting top 3 subnets by emission
    print("Top 3 subnets by emission:")
    emission_subnets = analyzer.get_top_subnets_by_emission(limit=3)
    for subnet in emission_subnets:
        print(f"Subnet {subnet['netuid']}: emission = {subnet.get('emission', 0)}")
    
    # If there are top 3 subnets, get their metrics for comparison
    if emission_subnets:
        netuids = [s['netuid'] for s in emission_subnets]
        print("\nComparing subnet metrics:")
        metrics_df = analyzer.compare_subnet_metrics(netuids)
        print(metrics_df)
        
        # Compare price predictions
        print("\nComparing price predictions:")
        price_comparison = analyzer.compare_price_predictions(netuids, days=15)
        if price_comparison['success']:
            analyzer.visualize_price_comparison(price_comparison, show_history=True)
    
    # Test investment recommendations
    print("\nGenerating investment recommendations:")
    recommendations = analyzer.generate_investment_recommendations(limit=3)
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. Subnet {rec['netuid']} - Score: {rec['investment_score']:.2f}")
        print(f"   Reason: {rec['recommendation_reason']}")
        print(f"   Current Price: {rec.get('price', 0):.6f} TAO")
        print(f"   Predicted Change: {rec.get('price_change_percent', 0):.2f}%") 
