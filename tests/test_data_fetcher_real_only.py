import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Keep tests isolated from bittensor runtime requirements.
fake_bt_module = types.ModuleType("bittensor")


class _FakeSubtensor:
    pass


def _fake_subtensor(*args, **kwargs):
    return _FakeSubtensor()


fake_bt_module.subtensor = _fake_subtensor
sys.modules["bittensor"] = fake_bt_module

from src.data_fetcher import DataFetcher


class DataFetcherRealOnlyTests(unittest.TestCase):
    @staticmethod
    def _response(status_code: int, payload=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = payload if payload is not None else {}
        resp.headers = {}
        return resp

    def test_real_only_raises_when_historical_api_fails(self):
        fetcher = DataFetcher(allow_mock_fallback=False)
        with patch("src.data_fetcher.requests.get", return_value=self._response(500)):
            with self.assertRaises(RuntimeError) as ctx:
                fetcher.get_historical_dtao_prices(netuid=1, days=30)

        self.assertIn("Mock fallback is disabled", str(ctx.exception))

    def test_mock_fallback_enabled_returns_generated_data(self):
        fetcher = DataFetcher(allow_mock_fallback=True)
        fallback_df = pd.DataFrame(
            {"price": [1.0]},
            index=pd.to_datetime(["2025-01-01"]),
        )

        with patch("src.data_fetcher.requests.get", return_value=self._response(500)):
            with patch.object(fetcher, "_generate_mock_price_data", return_value=fallback_df) as mock_generator:
                result = fetcher.get_historical_dtao_prices(netuid=1, days=30)

        mock_generator.assert_called_once_with(1, 30)
        pd.testing.assert_frame_equal(result, fallback_df)

    def test_env_var_enables_mock_fallback(self):
        with patch.dict(os.environ, {"DTAO_ALLOW_MOCK_DATA_FALLBACK": "1"}, clear=False):
            fetcher = DataFetcher()
            self.assertTrue(fetcher.allow_mock_fallback)

    def test_historical_pool_payload_is_parsed(self):
        fetcher = DataFetcher(allow_mock_fallback=False)
        payload = {
            "data": [
                {
                    "timestamp": "2025-01-01T00:00:00Z",
                    "price": "1.23",
                    "tao_volume_24_hr": "456.0",
                    "market_cap": "789.0",
                },
                {
                    "timestamp": "2025-01-02T00:00:00Z",
                    "price": "1.30",
                    "tao_volume_24_hr": "460.0",
                    "market_cap": "800.0",
                },
            ],
            "pagination": {"next_page": None},
        }
        with patch("src.data_fetcher.requests.get", return_value=self._response(200, payload)):
            result = fetcher.get_historical_dtao_prices(netuid=1, days=30)

        self.assertEqual(len(result), 2)
        self.assertIn("price", result.columns)
        self.assertIn("volume", result.columns)
        self.assertIn("market_cap", result.columns)
        self.assertAlmostEqual(float(result["price"].iloc[-1]), 1.30, places=6)

    def test_historical_prices_use_cache(self):
        fetcher = DataFetcher(allow_mock_fallback=False)
        payload = {
            "data": [
                {"timestamp": "2025-01-01T00:00:00Z", "price": "1.00"},
                {"timestamp": "2025-01-02T00:00:00Z", "price": "1.10"},
            ],
            "pagination": {"next_page": None},
        }

        with patch("src.data_fetcher.requests.get", return_value=self._response(200, payload)) as request_mock:
            first = fetcher.get_historical_dtao_prices(netuid=1, days=2)
            second = fetcher.get_historical_dtao_prices(netuid=1, days=2)

        self.assertEqual(request_mock.call_count, 1)
        pd.testing.assert_frame_equal(first, second)

    def test_taostats_price_endpoint_uses_pool_latest(self):
        fetcher = DataFetcher(allow_mock_fallback=False)
        payload = {
            "data": [
                {
                    "netuid": 1,
                    "price": "2.50",
                }
            ]
        }
        with patch("src.data_fetcher.requests.get", return_value=self._response(200, payload)) as request_mock:
            price = fetcher._get_taostats_subnet_price(netuid=1)

        self.assertAlmostEqual(price, 2.5, places=6)
        called_url = request_mock.call_args.kwargs.get("url") or request_mock.call_args.args[0]
        self.assertIn("/api/dtao/pool/latest/v1", called_url)

    def test_authorization_header_uses_raw_token(self):
        with patch("src.data_fetcher.TAOSTATS_API_KEY", "abc:def"):
            fetcher = DataFetcher(allow_mock_fallback=False)
            self.assertEqual(fetcher.headers.get("Authorization"), "abc:def")

    def test_authorization_header_strips_bearer_prefix(self):
        with patch("src.data_fetcher.TAOSTATS_API_KEY", "Bearer abc:def"):
            fetcher = DataFetcher(allow_mock_fallback=False)
            self.assertEqual(fetcher.headers.get("Authorization"), "abc:def")

    def test_taostats_get_retries_on_429(self):
        fetcher = DataFetcher(allow_mock_fallback=False)
        limited = self._response(429, {"message": "Rate Limited"})
        ok = self._response(200, {"data": []})

        with patch("src.data_fetcher.requests.get", side_effect=[limited, ok]) as request_mock:
            with patch("src.data_fetcher.time.sleep") as sleep_mock:
                response = fetcher._taostats_get("/api/dtao/pool/latest/v1", params={"limit": 1})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(request_mock.call_count, 2)
        sleep_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
