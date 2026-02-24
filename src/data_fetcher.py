import bittensor as bt
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Dict, List, Optional, Any
import random

# Add src directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SUBTENSOR_NETWORK, TAOSTATS_API_URL, TAOSTATS_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Data fetching class, responsible for getting data related to a subnet from Bittensor and Taostats
    """
    
    def __init__(self, allow_mock_fallback: Optional[bool] = None):
        self.subtensor = self._create_subtensor_client()
        self.headers = {"Content-Type": "application/json"}
        if TAOSTATS_API_KEY and TAOSTATS_API_KEY != "your_api_key":
            self.headers["Authorization"] = f"Bearer {TAOSTATS_API_KEY}"
        env_allow_mock = os.getenv("DTAO_ALLOW_MOCK_DATA_FALLBACK")
        if allow_mock_fallback is None:
            self.allow_mock_fallback = env_allow_mock == "1"
        else:
            self.allow_mock_fallback = allow_mock_fallback

    def _create_subtensor_client(self):
        """Create a bittensor Subtensor client compatible with old/new API naming."""
        if hasattr(bt, "subtensor"):
            return bt.subtensor(network=SUBTENSOR_NETWORK)
        if hasattr(bt, "Subtensor"):
            return bt.Subtensor(network=SUBTENSOR_NETWORK)
        raise RuntimeError("Unsupported bittensor version: no Subtensor client found")

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _extract_data_rows(payload: Any) -> List[Dict[str, Any]]:
        """Normalize Taostats payloads that may return list or {data: [...]}."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                return [data]
        return []

    @staticmethod
    def _response_error_message(response: requests.Response) -> str:
        """Best-effort extraction of API error details from HTTP responses."""
        message = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = str(payload.get("message", "")).strip()
        except Exception:
            message = ""
        if message:
            return f"Taostats HTTP {response.status_code}: {message}"
        return f"Taostats HTTP {response.status_code}"

    def _get_subnet_info(self, netuid: int):
        """Fetch subnet info object, handling API differences."""
        try:
            return self.subtensor.get_subnet_info(netuid=netuid)
        except TypeError:
            return self.subtensor.get_subnet_info(netuid)

    def _get_dynamic_subnet(self, netuid: int):
        """Fetch dynamic subnet object when available (newer bittensor API)."""
        if hasattr(self.subtensor, "subnet"):
            try:
                return self.subtensor.subnet(netuid)
            except Exception:
                return None
        return None

    def _get_all_subnet_ids(self) -> List[int]:
        """Fetch all subnet IDs, compatible across bittensor versions."""
        if hasattr(self.subtensor, "get_subnets"):
            return [int(uid) for uid in self.subtensor.get_subnets()]
        if hasattr(self.subtensor, "get_all_subnets_netuid"):
            return [int(uid) for uid in self.subtensor.get_all_subnets_netuid()]
        if hasattr(self.subtensor, "all_subnets"):
            all_subnets = self.subtensor.all_subnets()
            if isinstance(all_subnets, list):
                return [int(getattr(s, "netuid", s)) for s in all_subnets]
        raise RuntimeError("Subtensor client does not expose a subnet listing method")
    
    def get_subnets_list(self) -> List[Dict[str, Any]]:
        """Get a list of all subnets and their basic information"""
        try:
            subnet_info = []
            subnet_list = self._get_all_subnet_ids()
            
            for netuid in subnet_list:
                try:
                    subnet_data = self._get_subnet_info(netuid)
                    dynamic = self._get_dynamic_subnet(netuid)
                    emission = self._to_float(getattr(subnet_data, 'emission_value', 0.0), 0.0)
                    if emission <= 0 and dynamic is not None:
                        emission = self._to_float(getattr(dynamic, 'tao_in_emission', getattr(dynamic, 'emission', 0.0)), 0.0)
                    subnet_info.append({
                        'netuid': netuid,
                        'name': f"Subnet {netuid}",
                        'emission': emission,
                        'max_n': int(subnet_data.max_n) if hasattr(subnet_data, 'max_n') else 0,
                        'min_stake': self._to_float(getattr(subnet_data, 'min_stake', 0.0), 0.0),
                    })
                except Exception as subnet_error:
                    logger.error(f"Failed to get subnet {netuid} information: {str(subnet_error)}")
            
            return subnet_info
        
        except Exception as e:
            logger.error(f"Failed to get subnet list: {str(e)}")
            return []
    
    def get_subnet_dtao_price(self, netuid: int) -> Optional[float]:
        """Get the current dTAO price for a specific subnet"""
        try:
            price = None

            # Newer bittensor API: direct subnet price endpoint.
            if hasattr(self.subtensor, 'get_subnet_price'):
                try:
                    raw_price = self.subtensor.get_subnet_price(netuid)
                    price_val = self._to_float(raw_price, 0.0)
                    if price_val > 0:
                        price = price_val
                except Exception as price_error:
                    logger.warning(f"Failed direct subnet price lookup for subnet {netuid}: {str(price_error)}")

            # Newer bittensor API: dynamic subnet object includes price.
            if price is None:
                dynamic = self._get_dynamic_subnet(netuid)
                if dynamic is not None and hasattr(dynamic, 'price'):
                    dynamic_price = self._to_float(getattr(dynamic, 'price', 0.0), 0.0)
                    if dynamic_price > 0:
                        price = dynamic_price

            # Legacy fallback: calculate price via tau_in / alpha_in.
            if price is None:
                subnet_data = self._get_subnet_info(netuid)
                tau_in = self._to_float(getattr(subnet_data, 'tau_in', 0.0), 0.0)
                alpha_in = self._to_float(getattr(subnet_data, 'alpha_in', 0.0), 0.0)
                if alpha_in > 0:
                    calc_price = tau_in / alpha_in
                    if calc_price > 0:
                        price = calc_price
            
            # If unable to get directly from Bittensor, try Taostats
            if price is None or price <= 0:
                try:
                    price = self._get_taostats_subnet_price(netuid)
                except Exception as e:
                    logger.warning(f"Failed to get price from Taostats: {str(e)}")
            
            return price
        
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} dTAO price: {str(e)}")
            return None
    
    def _get_taostats_subnet_price(self, netuid: int) -> Optional[float]:
        """Get the dTAO price for a specific subnet from Taostats API"""
        try:
            url = f"{TAOSTATS_API_URL}/api/dtao/pool/latest/v1"
            params = {"netuid": netuid, "limit": 1}
            response = requests.get(url, headers=self.headers, params=params, timeout=20)
            
            if response.status_code == 200:
                rows = self._extract_data_rows(response.json())
                if rows:
                    price = self._to_float(rows[0].get('price'), 0.0)
                    if price > 0:
                        return price
            
            logger.warning(
                "Taostats API did not return subnet %s price data (%s)",
                netuid,
                self._response_error_message(response),
            )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} price from Taostats: {str(e)}")
            return None

    def _handle_historical_price_failure(self, netuid: int, days: int, reason: str) -> pd.DataFrame:
        """Handle missing real historical price data based on fallback policy."""
        cleaned_reason = reason.strip().rstrip(".")
        if self.allow_mock_fallback:
            logger.warning(
                "Historical price fetch failed for subnet %s (%s). "
                "Using mock fallback because DTAO_ALLOW_MOCK_DATA_FALLBACK=1.",
                netuid,
                cleaned_reason,
            )
            return self._generate_mock_price_data(netuid, days)

        raise RuntimeError(
            f"Failed to get real historical prices for subnet {netuid}: {cleaned_reason}. "
            "Mock fallback is disabled. Set DTAO_ALLOW_MOCK_DATA_FALLBACK=1 to opt in."
        )
    
    def get_historical_dtao_prices(self, netuid: int, days: int = 30) -> pd.DataFrame:
        """Get historical dTAO price data for a specific subnet over a period of time"""
        try:
            # Calculate start date from current date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            url = f"{TAOSTATS_API_URL}/api/dtao/pool/history/v1"

            # Use timestamp range against current Taostats history endpoint.
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())

            rows: List[Dict[str, Any]] = []
            page = 1
            max_pages = max(1, (days // 200) + 3)

            for _ in range(max_pages):
                params = {
                    "netuid": netuid,
                    "frequency": "by_day",
                    "timestamp_start": start_ts,
                    "timestamp_end": end_ts,
                    "page": page,
                    "limit": 200,
                }
                response = requests.get(url, headers=self.headers, params=params, timeout=20)
                if response.status_code != 200:
                    return self._handle_historical_price_failure(
                        netuid,
                        days,
                        self._response_error_message(response),
                    )

                payload = response.json()
                page_rows = self._extract_data_rows(payload)
                if not page_rows:
                    break

                rows.extend(page_rows)

                pagination = payload.get("pagination", {}) if isinstance(payload, dict) else {}
                next_page = pagination.get("next_page")
                if next_page in (None, "", 0):
                    break
                if isinstance(next_page, int):
                    page = next_page
                else:
                    page += 1

            if not rows:
                return self._handle_historical_price_failure(
                    netuid,
                    days,
                    "Taostats returned empty historical prices",
                )

            parsed_rows: List[Dict[str, Any]] = []
            for item in rows:
                timestamp = item.get("timestamp")
                price = item.get("price")
                if timestamp is None or price is None:
                    continue

                parsed_item: Dict[str, Any] = {
                    "date": pd.to_datetime(timestamp, utc=True, errors="coerce"),
                    "price": self._to_float(price, np.nan),
                }

                if item.get("tao_volume_24_hr") is not None:
                    parsed_item["volume"] = self._to_float(item.get("tao_volume_24_hr"), np.nan)
                if item.get("market_cap") is not None:
                    parsed_item["market_cap"] = self._to_float(item.get("market_cap"), np.nan)

                parsed_rows.append(parsed_item)

            if not parsed_rows:
                return self._handle_historical_price_failure(
                    netuid,
                    days,
                    "Taostats historical payload did not contain usable price rows",
                )

            df = pd.DataFrame(parsed_rows)
            df = df.dropna(subset=["date", "price"])
            if df.empty:
                return self._handle_historical_price_failure(
                    netuid,
                    days,
                    "Taostats historical payload contained only invalid dates/prices",
                )

            # Convert timestamps to naive datetime index for downstream consistency.
            if getattr(df["date"].dt, "tz", None) is not None:
                df["date"] = df["date"].dt.tz_convert(None)

            for optional_col in ("volume", "market_cap"):
                if optional_col in df.columns:
                    df[optional_col] = df[optional_col].replace([np.inf, -np.inf], np.nan).ffill().bfill()
                    if df[optional_col].isna().all():
                        df = df.drop(columns=[optional_col])

            df = (
                df.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")
            )

            return df.tail(days)
            
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} historical prices: {str(e)}")
            return self._handle_historical_price_failure(netuid, days, str(e))
    
    def _generate_mock_price_data(self, netuid: int, days: int) -> pd.DataFrame:
        """Generate mock price data for testing"""
        logger.warning(f"Generating mock price data for subnet {netuid}")
        
        # Get current price as baseline, if unavailable use random value
        current_price = self.get_subnet_dtao_price(netuid) or random.uniform(0.1, 10)
        
        # Generate past N days dates
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, 0, -1)]
        
        # Generate mock price data with some random fluctuations and slight upward trend
        # Base price fluctuates between 80%-120% of current price
        base_prices = [current_price * random.uniform(0.8, 1.2) for _ in range(days)]
        
        # Add slight trend
        for i in range(1, days):
            base_prices[i] = base_prices[i-1] * random.uniform(0.98, 1.03)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'price': base_prices,
            'volume': np.random.randint(1000, 10000, size=days),
            'market_cap': [p * np.random.randint(10000, 100000) for p in base_prices]
        })
        
        df.set_index('date', inplace=True)
        return df
    
    def get_subnet_metrics(self, netuid: int) -> Dict[str, Any]:
        """Get subnet-related metrics, including liquidity, validator count, miner count, etc."""
        try:
            subnet_data = self._get_subnet_info(netuid)
            dynamic = self._get_dynamic_subnet(netuid)
            emission = self._to_float(getattr(subnet_data, 'emission_value', 0.0), 0.0)
            if emission <= 0 and dynamic is not None:
                emission = self._to_float(getattr(dynamic, 'tao_in_emission', getattr(dynamic, 'emission', 0.0)), 0.0)
            
            metrics = {
                'netuid': netuid,
                'tau_in': (
                    self._to_float(getattr(dynamic, 'tao_in', 0.0), 0.0)
                    if dynamic is not None else
                    self._to_float(getattr(subnet_data, 'tau_in', 0.0), 0.0)
                ),
                'alpha_in': (
                    self._to_float(getattr(dynamic, 'alpha_in', 0.0), 0.0)
                    if dynamic is not None else
                    self._to_float(getattr(subnet_data, 'alpha_in', 0.0), 0.0)
                ),
                'alpha_out': (
                    self._to_float(getattr(dynamic, 'alpha_out', 0.0), 0.0)
                    if dynamic is not None else
                    self._to_float(getattr(subnet_data, 'alpha_out', 0.0), 0.0)
                ),
                'total_stake': self._to_float(getattr(subnet_data, 'total_stake', 0.0), 0.0),
                'emission': emission,
                'active_validators': 0,
                'active_miners': 0,
                'tempo': int(getattr(subnet_data, 'tempo', getattr(dynamic, 'tempo', 0)) or 0),
            }
            
            # Get active validators and miners count
            try:
                metagraph = self.subtensor.metagraph(netuid)
                metrics['active_validators'] = len([uid for uid in range(metagraph.n.item()) if metagraph.validator_permit[uid]])
                metrics['active_miners'] = len([uid for uid in range(metagraph.n.item()) if not metagraph.validator_permit[uid]])
            except Exception as mg_error:
                logger.error(f"Failed to get metagraph for subnet {netuid}: {str(mg_error)}")
            
            # Try to get additional metrics from Taostats
            try:
                taostats_metrics = self._get_taostats_subnet_metrics(netuid)
                if taostats_metrics:
                    metrics.update(taostats_metrics)
            except Exception as ts_error:
                logger.error(f"Failed to get additional metrics from Taostats: {str(ts_error)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} metrics: {str(e)}")
            return {
                'netuid': netuid,
                'error': str(e)
            }
    
    def _get_taostats_subnet_metrics(self, netuid: int) -> Dict[str, Any]:
        """Get additional subnet metrics from Taostats API"""
        try:
            url = f"{TAOSTATS_API_URL}/api/dtao/pool/latest/v1"
            params = {"netuid": netuid, "limit": 1}
            response = requests.get(url, headers=self.headers, params=params, timeout=20)
            
            if response.status_code == 200:
                rows = self._extract_data_rows(response.json())
                if rows:
                    row = rows[0]
                    return {
                        "price": self._to_float(row.get("price"), 0.0),
                        "market_cap": self._to_float(row.get("market_cap"), 0.0),
                        "liquidity": self._to_float(row.get("liquidity"), 0.0),
                        "tao_volume_24_hr": self._to_float(row.get("tao_volume_24_hr"), 0.0),
                        "price_change_1_day": self._to_float(row.get("price_change_1_day"), 0.0),
                        "rank": int(row.get("rank", 0) or 0),
                    }
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to get additional metrics from Taostats: {str(e)}")
            return {}


# Test code
if __name__ == "__main__":
    data_fetcher = DataFetcher()
    
    # Test getting subnet list
    subnets = data_fetcher.get_subnets_list()
    print(f"Found {len(subnets)} subnets")
    
    # Test getting first subnet info
    if subnets:
        first_subnet = subnets[0]['netuid']
        print(f"\nGetting info for subnet {first_subnet}")
        
        # Get current price
        price = data_fetcher.get_subnet_dtao_price(first_subnet)
        print(f"Current price: {price}")
        
        # Get historical price data
        history = data_fetcher.get_historical_dtao_prices(first_subnet, days=30)
        print(f"Historical data points: {len(history)}")
        
        # Get subnet metrics
        metrics = data_fetcher.get_subnet_metrics(first_subnet)
        print(f"Subnet metrics: {metrics}") 
