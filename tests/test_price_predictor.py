import unittest
from typing import Optional
import sys
import types

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Keep tests isolated from bittensor runtime requirements in src.data_fetcher.
fake_data_fetcher_module = types.ModuleType('src.data_fetcher')
fake_data_fetcher_module.DataFetcher = object
sys.modules.setdefault('src.data_fetcher', fake_data_fetcher_module)

from src.price_predictor import PricePredictor


class StubDataFetcher:
    def __init__(self, train_df: pd.DataFrame, predict_df: Optional[pd.DataFrame] = None):
        self.train_df = train_df
        self.predict_df = predict_df if predict_df is not None else train_df
        self.calls = 0

    def get_historical_dtao_prices(self, netuid: int, days: int) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            return self.train_df.copy()
        return self.predict_df.copy()


def build_price_df(days: int, include_volume: bool) -> pd.DataFrame:
    dates = pd.date_range('2025-01-01', periods=days, freq='D')
    prices = np.linspace(1.0, 2.0, days) + 0.03 * np.sin(np.arange(days))
    data = {'price': prices}

    if include_volume:
        data['volume'] = np.linspace(1000.0, 1200.0, days) + 10 * np.cos(np.arange(days))

    return pd.DataFrame(data, index=dates)


class PricePredictorFeatureSchemaTests(unittest.TestCase):
    def test_train_model_none_model_name_uses_default_without_warning(self):
        train_df = build_price_df(days=90, include_volume=True)
        predictor = PricePredictor(StubDataFetcher(train_df))

        with self.assertNoLogs('src.price_predictor', level='WARNING'):
            result = predictor.train_model(netuid=1, model_name=None, historical_days=90)

        self.assertTrue(result['success'], result.get('error'))
        self.assertEqual(result['model_name'], predictor.default_model)

    def test_train_model_persists_feature_columns(self):
        train_df = build_price_df(days=90, include_volume=True)
        predictor = PricePredictor(StubDataFetcher(train_df))

        result = predictor.train_model(netuid=1, model_name='linear', historical_days=90)

        self.assertTrue(result['success'], result.get('error'))
        self.assertIn('feature_columns', result)
        self.assertIsInstance(result['feature_columns'], list)
        self.assertGreater(len(result['feature_columns']), 0)
        self.assertEqual(len(result['feature_columns']), result['scaler'].n_features_in_)

    def test_predict_future_prices_reindexes_to_training_schema(self):
        train_df = build_price_df(days=90, include_volume=True)
        predict_df = build_price_df(days=35, include_volume=False)
        predictor = PricePredictor(StubDataFetcher(train_df, predict_df))

        result = predictor.predict_future_prices(netuid=1, days=5, model_name='linear')

        self.assertTrue(result['success'], result.get('error'))
        self.assertEqual(len(result['prediction']), 5)
        self.assertIn('predicted_price', result['prediction'].columns)

    def test_predict_future_prices_falls_back_when_schema_missing(self):
        train_df = build_price_df(days=90, include_volume=True)
        predict_df = build_price_df(days=35, include_volume=False)
        predictor = PricePredictor(StubDataFetcher(train_df, predict_df))

        X_train, y_train, _ = predictor.prepare_features(train_df, return_feature_columns=True)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        legacy_result = {
            'netuid': 1,
            'model_name': 'linear',
            'success': True,
            'metrics': {
                'cv_mse': 0.0,
                'train_mse': 0.0,
                'train_mae': 0.0,
                'train_r2': 1.0,
            },
            'model': model,
            'scaler': scaler,
        }
        predictor.train_model = lambda netuid, model_name=None, historical_days=60: legacy_result

        with self.assertLogs('src.price_predictor', level='INFO') as logs:
            result = predictor.predict_future_prices(netuid=1, days=5, model_name='linear')

        self.assertTrue(result['success'], result.get('error'))
        self.assertTrue(
            any('Fallback feature alignment' in line for line in logs.output),
            logs.output
        )

    def test_train_model_invalid_model_name_warns_and_falls_back(self):
        train_df = build_price_df(days=90, include_volume=True)
        predictor = PricePredictor(StubDataFetcher(train_df))

        with self.assertLogs('src.price_predictor', level='WARNING') as logs:
            result = predictor.train_model(netuid=1, model_name='invalid_model', historical_days=90)

        self.assertTrue(result['success'], result.get('error'))
        self.assertEqual(result['model_name'], predictor.default_model)
        self.assertTrue(
            any('Invalid model name' in line for line in logs.output),
            logs.output
        )


if __name__ == '__main__':
    unittest.main()
