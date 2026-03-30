import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── BUG FIX: Mock py_clob_client at the sys.modules level BEFORE any test
#    file imports execution_client or order_manager.  This allows CI to run
#    without py_clob_client installed (it's a heavy native dependency that
#    is only needed at runtime, not for unit tests that mock everything). ──

if "py_clob_client" not in sys.modules:
    _mock_client_module = MagicMock()
    _mock_clob_types_module = MagicMock()
    _mock_order_builder_module = MagicMock()
    _mock_constants_module = MagicMock()

    _mock_client_module.ClobClient = MagicMock()
    _mock_clob_types_module.OrderArgs = MagicMock()
    _mock_clob_types_module.OrderType = SimpleNamespace(GTC="GTC", FOK="FOK", GTD="GTD")
    _mock_clob_types_module.MarketOrderArgs = MagicMock()
    _mock_clob_types_module.BalanceAllowanceParams = MagicMock()
    _mock_clob_types_module.AssetType = SimpleNamespace(COLLATERAL="COLLATERAL", CONDITIONAL="CONDITIONAL")
    _mock_clob_types_module.ApiCreds = MagicMock()
    _mock_constants_module.BUY = "BUY"
    _mock_constants_module.SELL = "SELL"

    sys.modules["py_clob_client"] = MagicMock()
    sys.modules["py_clob_client.client"] = _mock_client_module
    sys.modules["py_clob_client.clob_types"] = _mock_clob_types_module
    sys.modules["py_clob_client.order_builder"] = _mock_order_builder_module
    sys.modules["py_clob_client.order_builder.constants"] = _mock_constants_module

# Also mock optional heavy deps that transitively imported modules try to load.
# Without these, importing order_manager → execution_client → py_clob_client
# or supervisor → stable_baselines3 fails in CI.
for _mod_name in [
    "stable_baselines3",
    "stable_baselines3.common",
    "stable_baselines3.common.vec_env",
    "stable_baselines3.common.env_util",
    "sb3_contrib",
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "tensorboard",
    "streamlit",
    "streamlit_autorefresh",
    "websockets",
    "lightgbm",
    "catboost",
    "optuna",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
