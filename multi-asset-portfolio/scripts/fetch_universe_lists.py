#!/usr/bin/env python3
"""
Fetch Universe Lists - S&P 500, Nikkei 225, ETFs, Forex

WikipediaからS&P500と日経225の銘柄リストを取得し、
ETF・為替ペアと合わせてuniverse_full.yamlを生成する。

Usage:
    python scripts/fetch_universe_lists.py
    python scripts/fetch_universe_lists.py --output config/universe_full.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table["Symbol"].tolist()
        # Clean tickers (remove dots, etc.)
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"  Found {len(tickers)} S&P 500 tickers")
        return sorted(tickers)
    except Exception as e:
        print(f"  Warning: Failed to fetch S&P 500: {e}")
        return get_fallback_sp500()


def fetch_nikkei225_tickers() -> list[str]:
    """Fetch Nikkei 225 tickers from Wikipedia."""
    print("Fetching Nikkei 225 tickers from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/Nikkei_225"
        tables = pd.read_html(url)
        # Find the table with ticker codes
        for table in tables:
            if "Code" in table.columns or "Ticker" in table.columns:
                col = "Code" if "Code" in table.columns else "Ticker"
                tickers = table[col].astype(str).tolist()
                tickers = [t.strip() for t in tickers if t.strip().isdigit()]
                if len(tickers) > 100:
                    print(f"  Found {len(tickers)} Nikkei 225 tickers")
                    return sorted(tickers)
        # Alternative: try different column names
        for table in tables:
            for col in table.columns:
                if "code" in col.lower() or "ticker" in col.lower():
                    tickers = table[col].astype(str).tolist()
                    tickers = [t.strip() for t in tickers if t.strip().isdigit()]
                    if len(tickers) > 100:
                        print(f"  Found {len(tickers)} Nikkei 225 tickers")
                        return sorted(tickers)
        raise ValueError("Could not find Nikkei 225 ticker table")
    except Exception as e:
        print(f"  Warning: Failed to fetch Nikkei 225: {e}")
        return get_fallback_nikkei225()


def get_fallback_sp500() -> list[str]:
    """S&P 500 full ticker list (as of 2025)."""
    return [
        # A
        "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
        "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG",
        "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
        "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH",
        "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
        # B
        "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG",
        "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B",
        "BRO", "BSX", "BWA", "BX", "BXP",
        # C
        "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL",
        "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI",
        "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
        "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT",
        "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH",
        "CTVA", "CVS", "CVX",
        # D
        "D", "DAL", "DAY", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI",
        "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE",
        "DUK", "DVA", "DVN", "DXCM",
        # E
        "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN",
        "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES", "ESS",
        "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
        # F
        "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO",
        "FIS", "FITB", "FLT", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV",
        # G
        "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW",
        "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
        # H
        "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX",
        "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
        # I
        "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP",
        "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
        # J
        "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM",
        # K
        "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI",
        "KMX", "KO", "KR",
        # L
        "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT",
        "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV",
        # M
        "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT",
        "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST",
        "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI",
        "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU",
        # N
        "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW",
        "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA",
        # O
        "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY",
        # P
        "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG",
        "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR",
        "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR",
        "PXD",
        # Q
        "QCOM", "QRVO",
        # R
        "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP",
        "ROST", "RSG", "RTX",
        # S
        "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO",
        "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK",
        "SWKS", "SYF", "SYK", "SYY",
        # T
        "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT",
        "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA",
        "TSN", "TT", "TTWO", "TXN", "TXT", "TYL",
        # U
        "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
        # V
        "V", "VFC", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST",
        "VTR", "VTRS", "VZ",
        # W
        "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB",
        "WMT", "WRB", "WST", "WTW", "WY", "WYNN",
        # X
        "XEL", "XOM", "XYL",
        # Y
        "YUM",
        # Z
        "ZBH", "ZBRA", "ZTS",
    ]


def get_fallback_nikkei225() -> list[str]:
    """Fallback Nikkei 225 major tickers."""
    return [
        "1332", "1333", "1605", "1721", "1801", "1802", "1803", "1808", "1812", "1925",
        "1928", "1963", "2002", "2269", "2282", "2413", "2432", "2501", "2502", "2503",
        "2531", "2768", "2801", "2802", "2871", "2914", "3086", "3099", "3101", "3103",
        "3105", "3289", "3382", "3401", "3402", "3405", "3407", "3436", "3861", "3863",
        "4004", "4005", "4021", "4042", "4043", "4061", "4063", "4151", "4183", "4188",
        "4208", "4272", "4324", "4452", "4502", "4503", "4506", "4507", "4519", "4523",
        "4543", "4568", "4578", "4631", "4661", "4689", "4704", "4751", "4755", "4901",
        "4902", "4911", "5019", "5020", "5101", "5108", "5201", "5202", "5214", "5232",
        "5233", "5301", "5332", "5333", "5401", "5406", "5411", "5541", "5631", "5703",
        "5706", "5707", "5711", "5713", "5714", "5801", "5802", "5803", "5901", "6098",
        "6103", "6113", "6178", "6273", "6301", "6302", "6305", "6326", "6361", "6367",
        "6471", "6472", "6473", "6501", "6502", "6503", "6504", "6506", "6645", "6674",
        "6701", "6702", "6703", "6724", "6752", "6753", "6758", "6762", "6770", "6841",
        "6857", "6861", "6902", "6952", "6954", "6971", "6976", "6981", "6988", "7003",
        "7004", "7011", "7012", "7013", "7180", "7186", "7201", "7202", "7203", "7205",
        "7211", "7261", "7267", "7269", "7270", "7272", "7731", "7733", "7735", "7751",
        "7752", "7762", "7832", "7911", "7912", "7951", "8001", "8002", "8015", "8028",
        "8031", "8035", "8053", "8058", "8233", "8252", "8253", "8267", "8303", "8304",
        "8306", "8308", "8309", "8316", "8331", "8354", "8355", "8411", "8591", "8601",
        "8604", "8628", "8630", "8697", "8725", "8750", "8766", "8795", "8801", "8802",
        "8804", "8830", "9001", "9005", "9007", "9008", "9009", "9020", "9021", "9022",
        "9062", "9064", "9101", "9104", "9107", "9202", "9301", "9412", "9432", "9433",
        "9434", "9501", "9502", "9503", "9531", "9532", "9602", "9613", "9681", "9735",
        "9766", "9983", "9984",
    ]


def get_etf_universe() -> dict:
    """Define comprehensive ETF universe."""
    return {
        "equity_us": [
            "SPY", "QQQ", "IWM", "VTI", "VOO", "IVV", "DIA", "RSP",
            "VUG", "VTV", "IWF", "IWD", "IJH", "IJR", "MDY", "VB",
        ],
        "equity_sector": [
            "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
            "XLC", "VGT", "VHT", "VFH", "VIS", "VCR", "VDC", "VPU", "VAW", "VNQ",
        ],
        "bonds": [
            "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "JNK", "BND", "AGG",
            "VCIT", "VCSH", "GOVT", "MUB", "EMB", "BNDX", "IAGG",
        ],
        "commodity": [
            "GLD", "SLV", "IAU", "SGOL", "GDX", "GDXJ",
            "USO", "BNO", "UNG", "DBA", "DBC", "PDBC", "GSG",
        ],
        "international": [
            "EFA", "EEM", "VEU", "IEMG", "VWO", "IEFA", "ACWI", "ACWX",
            "VEA", "IXUS", "SCZ", "VSS", "FM", "VT",
        ],
        "real_estate": [
            "VNQ", "VNQI", "IYR", "SCHH", "RWR", "REET", "REM", "MORT",
        ],
        "thematic": [
            "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ",
            "ICLN", "TAN", "QCLN", "LIT", "BOTZ", "ROBO",
            "HACK", "SKYY", "CLOU", "WCLD", "IGV",
        ],
        "volatility": [
            "VXX", "UVXY", "SVXY", "VIXY",
        ],
        "leveraged": [
            "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA",
            "SOXL", "SOXS", "LABU", "LABD", "FAS", "FAZ",
        ],
    }


def get_forex_pairs() -> dict:
    """Define forex pairs."""
    return {
        "major": [
            "USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X",
            "USDCHF=X", "USDCAD=X", "NZDUSD=X",
        ],
        "cross": [
            "EURJPY=X", "GBPJPY=X", "EURGBP=X", "AUDJPY=X",
            "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
            "EURAUD=X", "EURCHF=X", "EURNZD=X", "EURCAD=X",
            "GBPAUD=X", "GBPCHF=X", "GBPCAD=X", "GBPNZD=X",
            "AUDNZD=X", "AUDCAD=X", "AUDCHF=X",
        ],
        "emerging": [
            "USDMXN=X", "USDZAR=X", "USDTRY=X", "USDBRL=X",
            "USDCNY=X", "USDINR=X", "USDKRW=X", "USDSGD=X",
            "USDHKD=X", "USDTHB=X", "USDMYR=X", "USDIDR=X",
        ],
    }


def get_crypto_pairs() -> list[str]:
    """Define cryptocurrency pairs."""
    return [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "MATIC-USD",
        "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD",
    ]


def generate_universe_yaml(
    sp500: list[str],
    nikkei225: list[str],
    etfs: dict,
    forex: dict,
    crypto: list[str],
    output_path: Path,
) -> None:
    """Generate universe_full.yaml file."""
    print(f"\nGenerating {output_path}...")

    universe = {
        "# Universe Full Configuration": None,
        "# Generated": datetime.now().isoformat(),
        "# Total assets": f"~{len(sp500) + len(nikkei225) + sum(len(v) for v in etfs.values()) + sum(len(v) for v in forex.values()) + len(crypto)}",
        "universe": {
            "us_stocks": {
                "enabled": True,
                "description": "S&P 500 constituents",
                "count": len(sp500),
                "tickers": sp500,
            },
            "japan_stocks": {
                "enabled": True,
                "description": "Nikkei 225 constituents",
                "suffix": ".T",
                "count": len(nikkei225),
                "tickers": nikkei225,
            },
            "etfs": {
                "enabled": True,
                "description": "Global ETFs by category",
                "categories": etfs,
            },
            "forex": {
                "enabled": True,
                "description": "Currency pairs",
                "pairs": forex,
            },
            "crypto": {
                "enabled": False,
                "description": "Cryptocurrency pairs (disabled by default)",
                "tickers": crypto,
            },
        },
        "presets": {
            "us_large_cap": {
                "description": "US large cap stocks only",
                "include": ["us_stocks"],
                "max_assets": 100,
            },
            "diversified_etf": {
                "description": "Diversified ETF portfolio",
                "include": ["etfs.equity_us", "etfs.bonds", "etfs.commodity", "etfs.international"],
            },
            "global_multi_asset": {
                "description": "Global multi-asset universe",
                "include": ["us_stocks", "etfs", "forex"],
                "exclude": ["etfs.leveraged", "etfs.volatility"],
            },
            "japan_focus": {
                "description": "Japan-focused universe",
                "include": ["japan_stocks", "etfs.international"],
            },
        },
        "filters": {
            "min_market_cap": None,
            "min_avg_volume": 100000,
            "min_price": 5.0,
            "max_spread_pct": 1.0,
            "exclude_otc": True,
            "exclude_adr": False,
        },
    }

    # Write YAML with custom formatting
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Universe Full Configuration\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total US Stocks: {len(sp500)}\n")
        f.write(f"# Total Japan Stocks: {len(nikkei225)}\n")
        f.write(f"# Total ETFs: {sum(len(v) for v in etfs.values())}\n")
        f.write(f"# Total Forex Pairs: {sum(len(v) for v in forex.values())}\n")
        f.write(f"# Total Crypto: {len(crypto)}\n")
        f.write("\n")

        # Remove comment keys before dumping
        del universe["# Universe Full Configuration"]
        del universe["# Generated"]
        del universe["# Total assets"]

        yaml.dump(universe, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Created {output_path}")
    print(f"  US Stocks: {len(sp500)}")
    print(f"  Japan Stocks: {len(nikkei225)}")
    print(f"  ETFs: {sum(len(v) for v in etfs.values())}")
    print(f"  Forex Pairs: {sum(len(v) for v in forex.values())}")
    print(f"  Crypto: {len(crypto)}")


def main():
    parser = argparse.ArgumentParser(description="Fetch universe lists and generate YAML")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("config/universe_full.yaml"),
        help="Output YAML file path",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use fallback lists (no network access)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Universe List Generator")
    print("=" * 60)

    # Fetch or use fallback lists
    if args.offline:
        print("\nUsing offline/fallback lists...")
        sp500 = get_fallback_sp500()
        nikkei225 = get_fallback_nikkei225()
    else:
        sp500 = fetch_sp500_tickers()
        nikkei225 = fetch_nikkei225_tickers()

    # Get static lists
    etfs = get_etf_universe()
    forex = get_forex_pairs()
    crypto = get_crypto_pairs()

    # Generate YAML
    generate_universe_yaml(sp500, nikkei225, etfs, forex, crypto, args.output)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
