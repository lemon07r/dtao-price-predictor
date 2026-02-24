import unittest
from unittest.mock import patch

import pandas as pd

from src.comparison_analyzer import ComparisonAnalyzer


class StubFetcher:
    def __init__(self, subnet_count=20):
        self.price_calls = 0
        self._subnets = [
            {"netuid": i, "name": f"Subnet {i}", "emission": float(i) / 100.0}
            for i in range(subnet_count)
        ]
        self._snapshot = {
            i: {
                "price": 1.0 + i * 0.01,
                "price_change_1_day": float(i),
                "price_change_1_week": float(i) * 0.5,
                "price_change_1_month": float(i) * 0.2,
                "liquidity": 1000.0 + i,
                "tao_volume_24_hr": 500.0 + i,
            }
            for i in range(subnet_count)
        }

    def get_subnets_list(self):
        return list(self._subnets)

    def get_taostats_pool_snapshot(self):
        return dict(self._snapshot)

    def get_subnet_dtao_price(self, netuid):
        self.price_calls += 1
        return self._snapshot.get(netuid, {}).get("price", 0.0)

    def get_subnet_metrics(self, netuid):
        subnet = next((s for s in self._subnets if s["netuid"] == netuid), None) or {}
        return {
            "netuid": netuid,
            "emission": subnet.get("emission", 0.0),
            "alpha_in": 10.0,
            "alpha_out": 2.0,
            "active_validators": 5,
            "active_miners": 20,
        }


class StubPredictor:
    def __init__(self):
        self.calls = 0

    def predict_future_prices(self, netuid, days=30, model_name=None):
        self.calls += 1
        prediction_df = pd.DataFrame({"predicted_price": [1.1]})
        return {
            "success": True,
            "current_price": 1.0,
            "prediction": prediction_df,
            "price_change_percent": 10.0,
            "metrics": {"train_r2": 0.9},
            "model_name": "random_forest",
        }


class ComparisonAnalyzerRateLimitTests(unittest.TestCase):
    def test_growth_predictions_are_limited_to_candidate_cap(self):
        fetcher = StubFetcher(subnet_count=30)
        predictor = StubPredictor()
        analyzer = ComparisonAnalyzer(fetcher, predictor)

        with patch("src.comparison_analyzer.ADVICE_PREDICTION_CANDIDATES", 4):
            result = analyzer.get_top_growth_subnets(days=30, limit=5)

        self.assertGreater(len(result), 0)
        self.assertLessEqual(predictor.calls, 4)

    def test_top_subnets_by_price_uses_snapshot_without_per_subnet_price_calls(self):
        fetcher = StubFetcher(subnet_count=20)
        predictor = StubPredictor()
        analyzer = ComparisonAnalyzer(fetcher, predictor)

        result = analyzer.get_top_subnets_by_price(limit=5)

        self.assertEqual(len(result), 5)
        self.assertEqual(fetcher.price_calls, 0)

    def test_generate_mining_recommendations_limits_predictions(self):
        fetcher = StubFetcher(subnet_count=30)
        predictor = StubPredictor()
        analyzer = ComparisonAnalyzer(fetcher, predictor)

        with patch("src.comparison_analyzer.ADVICE_PREDICTION_CANDIDATES", 4):
            result = analyzer.generate_mining_recommendations(
                limit=5,
                gpu_clusters=2.0,
                daily_cluster_cost_tao=0.0,
            )

        self.assertGreater(len(result), 0)
        self.assertLessEqual(predictor.calls, 4)
        self.assertIn("mining_profitability_score", result[0])
        self.assertIn("expected_miner_share_pct", result[0])


if __name__ == "__main__":
    unittest.main()
