import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import load_env_from_dotenv


class ConfigDotenvTests(unittest.TestCase):
    def test_load_env_from_dotenv_sets_values(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fp:
            fp.write(
                "# comment\n"
                "TAOSTATS_API_KEY=test_key\n"
                "TAOSTATS_API_URL=\"https://example.test\"\n"
                "export SUBTENSOR_NETWORK=local\n"
                "WITH_COMMENT=value # inline comment\n"
                "QUOTED_HASH=\"a # b\"\n"
            )
            dotenv_path = fp.name

        try:
            with patch.dict(os.environ, {}, clear=False):
                for key in (
                    "TAOSTATS_API_KEY",
                    "TAOSTATS_API_URL",
                    "SUBTENSOR_NETWORK",
                    "WITH_COMMENT",
                    "QUOTED_HASH",
                ):
                    os.environ.pop(key, None)

                load_env_from_dotenv(dotenv_path)

                self.assertEqual(os.environ.get("TAOSTATS_API_KEY"), "test_key")
                self.assertEqual(os.environ.get("TAOSTATS_API_URL"), "https://example.test")
                self.assertEqual(os.environ.get("SUBTENSOR_NETWORK"), "local")
                self.assertEqual(os.environ.get("WITH_COMMENT"), "value")
                self.assertEqual(os.environ.get("QUOTED_HASH"), "a # b")
        finally:
            Path(dotenv_path).unlink(missing_ok=True)

    def test_load_env_from_dotenv_does_not_override_existing(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fp:
            fp.write("TAOSTATS_API_KEY=file_value\n")
            dotenv_path = fp.name

        try:
            with patch.dict(os.environ, {"TAOSTATS_API_KEY": "preset_value"}, clear=False):
                load_env_from_dotenv(dotenv_path)
                self.assertEqual(os.environ.get("TAOSTATS_API_KEY"), "preset_value")
        finally:
            Path(dotenv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
