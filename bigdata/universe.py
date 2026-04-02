"""Universe definition for the portfolio risk engine."""

ASSET_CLASS_MAPPING = {
    "SPY": "Equities",
    "QQQ": "Equities",
    "IWM": "Equities",
    "TLT": "Fixed Income",
    "IEF": "Fixed Income",
    "GLD": "Commodities",
    "USO": "Commodities",
    "UUP": "FX Proxy",
    "EEM": "Equities",
}

ASSET_LIST = list(ASSET_CLASS_MAPPING.keys())

ASSET_UNIVERSE = [
    {"ticker": ticker, "asset_class": asset_class}
    for ticker, asset_class in ASSET_CLASS_MAPPING.items()
]
