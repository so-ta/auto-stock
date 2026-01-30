# TODO: 学術的シグナル実装計画

> **作成日**: 2026-01-30
> **目的**: 学術的に実績のあるシグナルの段階的実装

---

## 実装状況サマリー

| Phase | シグナル数 | データコスト | 状態 |
|-------|-----------|-------------|------|
| Phase 1 | 5 | 無料 | 🔥 実装中 (cmd_042) |
| Phase 2 | 5 | $19/月 | 📋 TODO |
| Phase 3 | 1 | $50-100/月 | 📋 TODO |

---

## Phase 1: 無料データシグナル（実装中）

**コマンド**: cmd_042
**状態**: 🔥 実装中

| シグナル | 学術的根拠 | データソース | 期待リターン |
|---------|-----------|-------------|-------------|
| Lead-Lag関係 | Oxford研究 | yfinance | 年率20%+ |
| 52週高値モメンタム | George & Hwang (2004) | yfinance | 従来モメンタム超過 |
| 短期リバーサル | Jegadeesh (1990) | yfinance | 月次0.5-1% |
| インサイダー取引 | Seyhun研究 | SEC EDGAR (無料) | 月次50bp+ |
| ショートインタレスト | Rapach et al. (2016) | FINRA (無料) | 年率14.6% |

---

## Phase 2: ファンダメンタルデータシグナル（TODO）

**必要データソース**: Financial Modeling Prep Starter ($19/月)
**優先度**: 高
**前提条件**: Phase 1完了後

### 2.1 PEAD (Post-Earnings Announcement Drift)

**学術的根拠**:
- Ball & Brown (1968): 最初の発見
- Bernard & Thomas (1989): 体系的研究
- 決算発表後60日間のドリフト

**必要データ**:
- 決算発表日
- EPS予想・実績
- サプライズ（実績 - 予想）

**実装概要**:
```python
class PEADSignal(SignalBase):
    """
    決算発表後ドリフトシグナル

    SUE (Standardized Unexpected Earnings) を計算し、
    ポジティブサプライズは買い、ネガティブは売り。
    """

    def compute(self, prices, earnings_data):
        # SUE = (Actual EPS - Expected EPS) / std(past surprises)
        # シグナル = SUE の符号 × 60日間の減衰関数
        pass
```

**パラメータ**:
- lookback_quarters: [4, 8, 12] (SUE標準化期間)
- decay_days: [30, 45, 60] (シグナル減衰期間)

---

### 2.2 Accruals Anomaly (会計発生高)

**学術的根拠**:
- Sloan (1996): 年率12%のヘッジリターン
- 高Accruals企業は将来の収益悪化傾向

**必要データ**:
- 財務諸表（四半期）
  - Total Assets
  - Cash & Cash Equivalents
  - Current Liabilities
  - Long-term Debt

**実装概要**:
```python
class AccrualsSignal(SignalBase):
    """
    会計発生高シグナル

    Accruals = (ΔCA - ΔCash) - (ΔCL - ΔSTD) - Depreciation
    高Accruals = 売り、低Accruals = 買い
    """

    def compute(self, prices, financials):
        # Accrual Ratio = Accruals / Total Assets
        # シグナル = -1 * Accrual Ratio (Zスコア正規化)
        pass
```

**パラメータ**:
- lookback_years: [1, 2, 3]
- use_ttm: [True, False] (Trailing Twelve Months)

---

### 2.3 Asset Growth Anomaly (資産成長)

**学術的根拠**:
- Cooper et al. (2008): 年率7.3%スプレッド
- 高成長企業は過大評価される傾向

**必要データ**:
- Total Assets（四半期/年次）

**実装概要**:
```python
class AssetGrowthSignal(SignalBase):
    """
    資産成長シグナル

    Asset Growth = (Total Assets_t / Total Assets_{t-1}) - 1
    高成長 = 売り、低成長 = 買い
    """

    def compute(self, prices, financials):
        # シグナル = -1 * Asset Growth (クロスセクショナルランキング)
        pass
```

**パラメータ**:
- lookback_years: [1, 2, 3]
- use_quarterly: [True, False]

---

### 2.4 Net Issuance (株式発行)

**学術的根拠**:
- Loughran & Ritter (1995): SEO後の低リターン
- Ikenberry et al. (1995): バイバック後の高リターン

**必要データ**:
- Shares Outstanding（四半期）
- Stock Split情報

**実装概要**:
```python
class NetIssuanceSignal(SignalBase):
    """
    株式発行シグナル

    Net Issuance = (Shares_t / Shares_{t-1}) - 1
    発行（増加） = 売り、バイバック（減少） = 買い
    """

    def compute(self, prices, shares_outstanding):
        # シグナル = -1 * Net Issuance
        pass
```

**パラメータ**:
- lookback_months: [3, 6, 12]
- exclude_splits: [True, False]

---

### 2.5 Gross Profitability (売上総利益率)

**学術的根拠**:
- Novy-Marx (2013): バリューと同等の予測力
- Journal of Financial Economics掲載

**必要データ**:
- Revenue (売上)
- Cost of Goods Sold (売上原価)
- Total Assets

**実装概要**:
```python
class GrossProfitabilitySignal(SignalBase):
    """
    売上総利益率シグナル

    GP/A = (Revenue - COGS) / Total Assets
    高収益性 = 買い
    """

    def compute(self, prices, financials):
        # シグナル = GP/A (クロスセクショナルランキング)
        pass
```

**パラメータ**:
- use_ttm: [True, False]
- sector_neutral: [True, False]

---

## Phase 3: 代替データシグナル（TODO）

**必要データソース**: IVolatility/ORATS ($50-100/月)
**優先度**: 中
**前提条件**: Phase 2完了後、効果検証

### 3.1 Option Implied Volatility Skew

**学術的根拠**:
- Xing et al. (2010): 年率10.9%リスク調整済みリターン
- インフォームドトレーダーはオプション市場で取引

**必要データ**:
- オプション価格（ATM/OTMプット・コール）
- インプライドボラティリティ

**実装概要**:
```python
class OptionSkewSignal(SignalBase):
    """
    オプションIVスキューシグナル

    Skew = IV(OTM Put) - IV(ATM Call)
    高スキュー（下落懸念） = 売り
    """

    def compute(self, prices, options_data):
        # シグナル = -1 * IV Skew
        pass
```

---

## データソース詳細

### 無料データソース

| ソース | データ | API |
|--------|-------|-----|
| Yahoo Finance (yfinance) | 価格・分割・配当 | 非公式Python API |
| SEC EDGAR | Form 4 (インサイダー) | 公式REST API |
| FINRA | ショートインタレスト | 公式REST API |

### 有料データソース

| ソース | 価格 | データ | 履歴 |
|--------|------|-------|------|
| Financial Modeling Prep | $19/月 | 財務諸表・決算 | 30年 |
| SimFin | €10-50/月 | 財務諸表・比率 | 20年 |
| Sharadar/Nasdaq | $50/月 | 全包括・高品質 | 30年 |
| IVolatility | $50-100/月 | オプションIV | 20年 |

---

## 実装優先度マトリックス

```
                    期待リターン
                    高
                    │
        Phase 1     │  Phase 2
        (無料)      │  ($19/月)
     ┌──────────────┼──────────────┐
     │  Lead-Lag   │  PEAD        │
     │  52W High   │  Accruals    │
     │  Reversal   │  AssetGrowth │
     │  Insider    │  NetIssuance │
     │  ShortInt   │  GrossProfit │
低コスト──────────────┼──────────────高コスト
     │              │              │
     │              │  Phase 3     │
     │              │  ($100/月)   │
     │              │  OptionSkew  │
     │              │              │
                    │
                    低
```

---

## 次のアクション

1. **即時**: cmd_042（Phase 1）の完了を待つ
2. **Phase 1完了後**: 効果検証バックテスト実施
3. **効果確認後**: Phase 2実装判断（$19/月の投資対効果）
4. **Phase 2完了後**: Phase 3検討

---

## 参考文献

### Phase 1
- George, T. J., & Hwang, C. Y. (2004). The 52-week high and momentum investing. *The Journal of Finance*, 59(5), 2145-2176.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *The Journal of Finance*, 45(3), 881-898.
- Seyhun, H. N. (1986). Insiders' profits, costs of trading, and market efficiency. *Journal of Financial Economics*, 16(2), 189-212.
- Rapach, D. E., Ringgenberg, M. C., & Zhou, G. (2016). Short interest and aggregate stock returns. *Journal of Financial Economics*, 121(1), 46-65.

### Phase 2
- Ball, R., & Brown, P. (1968). An empirical evaluation of accounting income numbers. *Journal of Accounting Research*, 6(2), 159-178.
- Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? *The Accounting Review*, 71(3), 289-315.
- Cooper, M. J., Gulen, H., & Schill, M. J. (2008). Asset growth and the cross-section of stock returns. *The Journal of Finance*, 63(4), 1609-1651.
- Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1-28.

### Phase 3
- Xing, Y., Zhang, X., & Zhao, R. (2010). What does the individual option volatility smirk tell us about future equity returns? *Journal of Financial and Quantitative Analysis*, 45(3), 641-662.
