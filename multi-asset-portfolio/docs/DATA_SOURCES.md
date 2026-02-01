# データソースとシグナル一覧

> **Last Updated**: 2026-02-01

本ドキュメントでは、multi-asset-portfolio で使用されているシグナルと、それらのデータソースについて説明する。

---

## 目次

1. [データソース概要](#1-データソース概要)
2. [シグナル一覧](#2-シグナル一覧)
3. [銘柄ユニバース](#3-銘柄ユニバース)
4. [データフロー](#4-データフロー)
5. [キャッシュシステム](#5-キャッシュシステム)

---

## 1. データソース概要

### 主要データソース

| データソース | 用途 | データ種別 | 認証 |
|-------------|------|-----------|------|
| **Yahoo Finance** | 株価・ETF価格 | OHLCV | 不要 |
| **FRED API** | マクロ経済指標 | 金利、インフレ等 | 不要 |
| **SEC EDGAR** | インサイダー取引 | Form 4 | 不要 (User-Agent必須) |
| **FINRA API** | 空売りデータ | Short Interest | 不要 |

### 1.1 Yahoo Finance

**主要用途**: 株価・ETF の OHLCV (始値・高値・安値・終値・出来高) データ

```python
# 実装: src/data/adapters/stock.py (StockAdapter)
import yfinance as yf
data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
```

**対応市場**:
- US 市場: `AAPL`, `MSFT` 等
- 日本市場: `.T` サフィックス (例: `7203.T`)
- 香港市場: `.HK` サフィックス
- 韓国市場: `.KS` サフィックス
- 英国市場: `.L` サフィックス

**設定**: `config/default.yaml`
```yaml
data:
  stock:
    use_adjusted_prices: true  # 株式分割・配当調整済み価格
```

### 1.2 FRED API (Federal Reserve Economic Data)

**主要用途**: マクロ経済指標、金利データ

```
シリーズコード:
- DGS3MO: 3ヶ月国債利回り
- DGS2:   2年国債利回り
- DGS5:   5年国債利回り
- DGS10:  10年国債利回り
- DGS30:  30年国債利回り
```

**実装**: `src/signals/yield_curve_signal.py` (EnhancedYieldCurveSignal)

**API**: https://fred.stlouisfed.org/

### 1.3 SEC EDGAR API

**主要用途**: インサイダー取引データ (Form 4)

```
取得データ:
- CIK → Ticker マッピング
- 取引日、取引種別、株数、価格
- Executive vs Non-executive 区別
- Direct vs Indirect 保有区別
```

**実装**: `src/data/sec_edgar.py` (SECEdgarClient)

**API**: https://www.sec.gov/developer

**制限**: 10リクエスト/秒推奨、User-Agent ヘッダ必須

### 1.4 FINRA API

**主要用途**: 空売り残高データ

```
取得データ:
- Short Interest (株数)
- Average Daily Volume
- Days to Cover = SI / ADV
```

**更新頻度**: 隔週 (月中・月末決済日)

**実装**: `src/data/finra.py` (FINRAClient)

---

## 2. シグナル一覧

全 **70+ シグナル** を **15 カテゴリ** に分類。

### 2.1 モメンタム (Momentum)

| シグナル | 入力データ | データソース |
|---------|-----------|-------------|
| `momentum_return` | close | Yahoo Finance |
| `roc` (Rate of Change) | close | Yahoo Finance |
| `momentum_composite` | close | Yahoo Finance |
| `momentum_acceleration` | close | Yahoo Finance |
| `fifty_two_week_high_momentum` | high, low, close | Yahoo Finance |

### 2.2 平均回帰 (Mean Reversion)

| シグナル | 入力データ | 計算手法 |
|---------|-----------|---------|
| `bollinger_reversion` | close | ボリンジャーバンド |
| `rsi` | close | RSI |
| `zscore_reversion` | close | Z-スコア |
| `stochastic_reversion` | high, low, close | Stochastic |

### 2.3 ボラティリティ (Volatility)

| シグナル | 入力データ | 計算手法 |
|---------|-----------|---------|
| `atr` | high, low, close | Average True Range |
| `volatility_breakout` | close | ボラティリティ調整ブレークアウト |
| `volatility_regime` | close | レジーム検出 |

### 2.4 ブレークアウト (Breakout)

| シグナル | 入力データ | 計算手法 |
|---------|-----------|---------|
| `donchian_channel` | high, low, close | Donchian Channel |
| `high_low_breakout` | high, low, close | 高値・安値ブレークアウト |
| `range_breakout` | OHLC | レンジブレークアウト |

### 2.5 ファクター (Factor)

| シグナル | 入力データ | 学術基盤 |
|---------|-----------|---------|
| `value_factor` | high, close | 52週高値乖離 |
| `quality_factor` | close | 安定性・一貫性 |
| `low_vol_factor` | close | 低ボラティリティプレミアム |
| `momentum_factor` | close | クロスセクション相対強度 |
| `size_factor` | close | 流動性・時価総額 |

### 2.6 セクター (Sector)

| シグナル | 入力データ | データソース |
|---------|-----------|-------------|
| `sector_momentum` | セクターETF | Yahoo Finance |
| `sector_relative_strength` | セクターETF | Yahoo Finance |
| `sector_breadth` | セクターETF | Yahoo Finance |

**使用セクターETF**:
```
XLK (Technology), XLF (Financials), XLE (Energy),
XLV (Healthcare), XLY (Consumer Discretionary),
XLP (Consumer Staples), XLI (Industrials),
XLB (Materials), XLU (Utilities), XLRE (Real Estate),
XLC (Communication Services)
```

### 2.7 ボリューム (Volume)

| シグナル | 入力データ | 計算手法 |
|---------|-----------|---------|
| `obv_momentum` | close, volume | On-Balance Volume |
| `money_flow_index` | HLCV | MFI (ボリューム加重RSI) |
| `vwap_deviation` | HLCV | VWAP乖離率 |
| `accumulation_distribution` | HLCV | A/Dライン |

### 2.8 センチメント (Sentiment)

| シグナル | 入力データ | データソース |
|---------|-----------|-------------|
| `vix_sentiment` | VIX指数 | Yahoo Finance (^VIX) |
| `put_call_ratio` | PCR指数 | Yahoo Finance |
| `market_breadth` | NYSE A/D | Yahoo Finance |
| `fear_greed_composite` | VIX + PCR + 他 | 複合指標 |
| `vix_term_structure` | VIX期間構造 | Yahoo Finance |

### 2.9 マクロ経済 (Macro)

| シグナル | 入力データ | データソース |
|---------|-----------|-------------|
| `yield_curve` | TLT/SHY比率 | Yahoo Finance |
| `inflation_expectations` | TIP/IEF比率 | Yahoo Finance |
| `credit_spread` | HYG/LQD比率 | Yahoo Finance |
| `dollar_strength` | DXY | Yahoo Finance |
| `macro_regime_composite` | 複合 | 複合指標 |
| `enhanced_yield_curve` | 金利データ | FRED API |

### 2.10 短期反発 (Short-Term Reversal)

| シグナル | 入力データ | 学術基盤 |
|---------|-----------|---------|
| `short_term_reversal` | close, volume | Jegadeesh (1990) |
| `weekly_short_term_reversal` | 週次リターン | 1週間反発効果 |
| `monthly_short_term_reversal` | 月次リターン | 1ヶ月反発効果 |

### 2.11 トレンド (Trend)

| シグナル | 入力データ | 計算手法 |
|---------|-----------|---------|
| `dual_momentum` | OHLCV + ベンチマーク | 絶対+相対モメンタム |
| `trend_following` | OHLCV | MA交差 + ADX + Donchian |
| `adaptive_trend` | OHLCV | ボラティリティ適応型 |
| `cross_asset_momentum` | 複数資産 | クロスアセット相対強度 |
| `multi_timeframe_momentum` | OHLCV | マルチタイムフレーム |
| `timeframe_consensus` | OHLCV | タイムフレーム間コンセンサス |

### 2.12 季節性 (Seasonality)

| シグナル | 入力データ | 注意事項 |
|---------|-----------|---------|
| `day_of_week` | close | オーバーフィッティング注意 |
| `month_effect` | close | 市場効率性で消滅傾向 |
| `turn_of_month` | close | 限定的証拠 |

### 2.13 インサイダー取引 (Insider Trading)

| シグナル | 入力データ | データソース |
|---------|-----------|-------------|
| `insider_trading` | SEC Form 4 | SEC EDGAR API |

**学術基盤**: Seyhun (1986, 1998)

### 2.14 空売り (Short Interest)

| シグナル | 入力データ | データソース |
|---------|-----------|-------------|
| `short_interest` | 空売り残高 | FINRA API |
| `short_interest_change` | 空売り変化 | FINRA API |

**学術基盤**: Rapach et al. (2016), Boehmer et al. (2008)

### 2.15 その他 (Advanced/Composite)

| シグナル | 入力データ | 用途 |
|---------|-----------|-----|
| `momentum_ensemble` | マルチタイムフレーム | 複合モメンタム |
| `mean_reversion_ensemble` | 複数指標 | 複合平均回帰 |
| `trend_strength` | OHLCV | トレンド強度 |
| `regime_detector` | OHLCV | レジーム分類 |
| `kama` | close | Kaufman適応型MA |
| `keltner_channel` | HLC | ATRベースチャネル |
| `correlation_regime` | OHLCV + リファレンス | 相関レジーム |
| `return_dispersion` | マルチアセット | リターン分散 |
| `lead_lag` | マルチアセット | 銘柄間時差相関 |

---

## 3. 銘柄ユニバース

### 3.1 Asset Master (config/asset_master.yaml)

銘柄ユニバースの Single Source of Truth。

**構造**:
```yaml
version: "3.0"

assets:
  AAPL:
    name: "Apple Inc."
    market: us
    asset_class: equity
    sector: technology
    tags: ["sp500", "quality", "sbi"]

  SPY:
    name: "SPDR S&P 500 ETF"
    market: us
    asset_class: etf
    etf_category: market_cap
    tags: ["benchmark", "sbi"]
```

### 3.2 対応市場

| 市場コード | 説明 | シンボル形式 |
|-----------|------|-------------|
| `us` | 米国市場 | `AAPL` |
| `japan` | 日本市場 | `7203.T` |
| `hong_kong` | 香港市場 | `0700.HK` |
| `korea` | 韓国市場 | `005930.KS` |
| `global` | グローバル | 各種 |

### 3.3 資産クラス

| クラス | 説明 |
|-------|------|
| `equity` | 個別株 |
| `etf` | ETF |
| `bond` | 債券 |
| `commodity` | コモディティ |
| `forex` | 外国為替 |
| `crypto` | 暗号資産 |

### 3.4 セクター分類 (GICS)

```
technology, healthcare, financials, industrials,
consumer_discretionary, consumer_staples, energy,
materials, utilities, real_estate, communication_services
```

---

## 4. データフロー

```
┌────────────────────────────────────────────────────────────┐
│                     データソース層                          │
├──────────┬──────────┬──────────┬──────────┬───────────────┤
│ Yahoo    │ FRED     │ SEC      │ FINRA    │ Asset Master  │
│ Finance  │ API      │ EDGAR    │ API      │ (YAML)        │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴───────┬───────┘
     │          │          │          │             │
     └──────────┴──────────┴──────────┴─────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ UnifiedCacheManager    │
              │ (S3 + Local Hybrid)    │
              └───────────┬────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
   ┌──────────┐    ┌───────────┐    ┌────────────┐
   │ Signal   │    │ Backtest  │    │ Covariance │
   │ Compute  │    │ Engine    │    │ Cache      │
   └────┬─────┘    └─────┬─────┘    └─────┬──────┘
        │                │                │
        └────────────────┴────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ Portfolio Optimization │
            │ (NCO, Risk Allocation) │
            └────────────────────────┘
```

---

## 5. キャッシュシステム

### 5.1 キャッシュタイプ

```python
from src.utils.cache_manager import CacheType

CacheType.SIGNAL        # シグナル計算結果
CacheType.DATAFRAME     # 汎用DataFrame
CacheType.DATA          # OHLCVデータ
CacheType.LRU           # 汎用LRU
CacheType.INCREMENTAL   # インクリメンタル計算
CacheType.COVARIANCE    # 共分散行列
```

### 5.2 ストレージ設定

```yaml
# config/default.yaml
storage:
  s3_bucket: "stock-local-dev-014498665038"
  s3_prefix: ".cache"
  s3_region: "ap-northeast-1"
  base_path: ".cache"
  local_cache_ttl_hours: 24
```

### 5.3 ローカルキャッシュ構造

```
.cache/
├── data/           # OHLCVデータ
├── signals/        # シグナル計算結果
├── covariance/     # 共分散行列
├── sec_edgar/      # SEC EDGARレスポンス
└── finra/          # FINRAレスポンス
```

詳細は [CACHE_SYSTEM.md](CACHE_SYSTEM.md) を参照。

---

## 関連ドキュメント

- [SIGNAL_PIPELINE.md](SIGNAL_PIPELINE.md) - シグナル追加手順
- [CACHE_SYSTEM.md](CACHE_SYSTEM.md) - キャッシュシステム詳細
- [ARCHITECTURE.md](ARCHITECTURE.md) - システムアーキテクチャ
