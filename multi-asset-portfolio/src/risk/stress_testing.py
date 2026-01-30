"""
Stress Testing Module - ストレステスト自動化

歴史的なストレスシナリオに対するポートフォリオの耐性をテストする。
主要な市場クラッシュシナリオを再現し、ポートフォリオ損失を評価。

主要コンポーネント:
- StressScenario: ストレスシナリオ定義
- HISTORICAL_SCENARIOS: 歴史的シナリオのプリセット
- StressTester: ストレステスト実行

設計根拠:
- 歴史的シナリオ: 実際に発生した市場イベントを再現
- 相関乗数: ストレス時の相関上昇を考慮
- ヘッジ提案: 最悪シナリオへの対策を提供

使用例:
    from src.risk.stress_testing import StressTester, HISTORICAL_SCENARIOS

    tester = StressTester()

    # 単一シナリオテスト
    result = tester.run_stress_test(
        weights={"SPY": 0.4, "TLT": 0.3, "GLD": 0.2, "VNQ": 0.1},
        scenario=HISTORICAL_SCENARIOS["GFC_2008"],
    )
    print(f"Portfolio loss: {result['portfolio_loss']:.2%}")

    # 全シナリオテスト
    results_df = tester.run_all_stress_tests(weights)
    print(results_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class StressScenario:
    """ストレスシナリオ定義

    Attributes:
        name: シナリオ名
        description: シナリオの説明
        shocks: 各資産のショック（例: {"SPY": -0.20} = SPY 20%下落）
        correlation_multiplier: ストレス時の相関乗数（1.0=通常、1.5=相関上昇）
        duration_days: シナリオの想定期間（日数）
        recovery_days: 回復に要した日数（歴史的参考）
    """
    name: str
    description: str
    shocks: Dict[str, float]
    correlation_multiplier: float = 1.5
    duration_days: int = 30
    recovery_days: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "shocks": self.shocks,
            "correlation_multiplier": self.correlation_multiplier,
            "duration_days": self.duration_days,
            "recovery_days": self.recovery_days,
        }

    def get_shock(self, asset: str, default: float = 0.0) -> float:
        """資産のショックを取得（デフォルト値付き）"""
        return self.shocks.get(asset, default)


@dataclass
class StressTestResult:
    """ストレステスト結果

    Attributes:
        scenario_name: シナリオ名
        portfolio_loss: ポートフォリオ損失率
        asset_impacts: 各資産の損失寄与
        survives: 生存判定（損失が閾値以下）
        weights: テストしたウェイト
        scenario: 使用したシナリオ
    """
    scenario_name: str
    portfolio_loss: float
    asset_impacts: Dict[str, float]
    survives: bool
    weights: Dict[str, float]
    scenario: StressScenario

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "portfolio_loss": self.portfolio_loss,
            "asset_impacts": self.asset_impacts,
            "survives": self.survives,
            "weights": self.weights,
        }


@dataclass
class HedgeRecommendation:
    """ヘッジ提案

    Attributes:
        hedge_asset: ヘッジ資産
        recommended_weight: 推奨ウェイト
        expected_protection: 期待される保護効果
        cost_estimate: コスト推定（年率）
        rationale: 推奨理由
    """
    hedge_asset: str
    recommended_weight: float
    expected_protection: float
    cost_estimate: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hedge_asset": self.hedge_asset,
            "recommended_weight": self.recommended_weight,
            "expected_protection": self.expected_protection,
            "cost_estimate": self.cost_estimate,
            "rationale": self.rationale,
        }


# =============================================================================
# 歴史的シナリオ定義
# =============================================================================

HISTORICAL_SCENARIOS: Dict[str, StressScenario] = {
    # 2008年金融危機（リーマンショック）
    "GFC_2008": StressScenario(
        name="GFC_2008",
        description="2008年グローバル金融危機（リーマンショック）",
        shocks={
            "SPY": -0.40,    # S&P 500: -40%
            "QQQ": -0.45,    # NASDAQ: -45%
            "TLT": 0.20,     # 長期国債: +20%（フライトトゥクオリティ）
            "GLD": 0.10,     # 金: +10%
            "VNQ": -0.50,    # 不動産: -50%
            "EFA": -0.45,    # 先進国株: -45%
            "EEM": -0.55,    # 新興国株: -55%
            "HYG": -0.25,    # ハイイールド債: -25%
            "LQD": -0.10,    # 投資適格社債: -10%
        },
        correlation_multiplier=1.8,
        duration_days=365,
        recovery_days=1100,
    ),

    # 2020年COVID-19クラッシュ
    "COVID_2020": StressScenario(
        name="COVID_2020",
        description="2020年COVID-19パンデミッククラッシュ",
        shocks={
            "SPY": -0.34,    # S&P 500: -34%
            "QQQ": -0.28,    # NASDAQ: -28%（テック比較的堅調）
            "TLT": 0.15,     # 長期国債: +15%
            "GLD": -0.05,    # 金: -5%（一時的流動性需要）
            "VNQ": -0.40,    # 不動産: -40%
            "USO": -0.70,    # 原油: -70%
            "EFA": -0.35,    # 先進国株: -35%
            "EEM": -0.35,    # 新興国株: -35%
            "HYG": -0.20,    # ハイイールド債: -20%
        },
        correlation_multiplier=1.6,
        duration_days=33,
        recovery_days=150,
    ),

    # 2022年利上げショック
    "Rate_Hike_2022": StressScenario(
        name="Rate_Hike_2022",
        description="2022年急速利上げによる株債券同時下落",
        shocks={
            "SPY": -0.25,    # S&P 500: -25%
            "QQQ": -0.33,    # NASDAQ: -33%（グロース株大打撃）
            "TLT": -0.30,    # 長期国債: -30%（債券暴落）
            "GLD": -0.05,    # 金: -5%
            "VNQ": -0.30,    # 不動産: -30%（金利感応）
            "EFA": -0.20,    # 先進国株: -20%
            "EEM": -0.25,    # 新興国株: -25%
            "HYG": -0.15,    # ハイイールド債: -15%
            "LQD": -0.20,    # 投資適格社債: -20%
        },
        correlation_multiplier=1.5,
        duration_days=270,
        recovery_days=None,  # まだ完全回復せず
    ),

    # フラッシュクラッシュ
    "Flash_Crash": StressScenario(
        name="Flash_Crash",
        description="短期的な急落（フラッシュクラッシュ型）",
        shocks={
            "SPY": -0.10,    # S&P 500: -10%
            "QQQ": -0.12,    # NASDAQ: -12%
            "TLT": 0.02,     # 長期国債: +2%
            "GLD": 0.01,     # 金: +1%
            "VNQ": -0.12,    # 不動産: -12%
            "EFA": -0.11,    # 先進国株: -11%
            "EEM": -0.13,    # 新興国株: -13%
        },
        correlation_multiplier=1.3,
        duration_days=1,
        recovery_days=5,
    ),

    # スタグフレーション
    "Stagflation": StressScenario(
        name="Stagflation",
        description="スタグフレーション（景気後退+高インフレ）",
        shocks={
            "SPY": -0.20,    # S&P 500: -20%
            "QQQ": -0.25,    # NASDAQ: -25%（グロース株不利）
            "TLT": -0.15,    # 長期国債: -15%（インフレで下落）
            "GLD": 0.15,     # 金: +15%（インフレヘッジ）
            "USO": 0.30,     # 原油: +30%（インフレ要因）
            "VNQ": -0.10,    # 不動産: -10%
            "EFA": -0.18,    # 先進国株: -18%
            "EEM": -0.15,    # 新興国株: -15%（資源国プラス）
            "TIP": 0.05,     # TIPS: +5%
        },
        correlation_multiplier=1.4,
        duration_days=365,
        recovery_days=730,
    ),

    # 新興国危機
    "EM_Crisis": StressScenario(
        name="EM_Crisis",
        description="新興国危機（通貨危機・債務危機）",
        shocks={
            "SPY": -0.15,    # S&P 500: -15%
            "QQQ": -0.18,    # NASDAQ: -18%
            "TLT": 0.10,     # 長期国債: +10%（安全資産）
            "GLD": 0.08,     # 金: +8%
            "EEM": -0.40,    # 新興国株: -40%
            "EFA": -0.20,    # 先進国株: -20%
            "HYG": -0.15,    # ハイイールド債: -15%
        },
        correlation_multiplier=1.5,
        duration_days=180,
        recovery_days=365,
    ),

    # テクノロジーバブル崩壊
    "Tech_Bubble": StressScenario(
        name="Tech_Bubble",
        description="テクノロジーバブル崩壊（2000年型）",
        shocks={
            "SPY": -0.25,    # S&P 500: -25%
            "QQQ": -0.50,    # NASDAQ: -50%（テック壊滅）
            "TLT": 0.15,     # 長期国債: +15%
            "GLD": 0.05,     # 金: +5%
            "VNQ": -0.10,    # 不動産: -10%
            "EFA": -0.30,    # 先進国株: -30%
        },
        correlation_multiplier=1.4,
        duration_days=730,
        recovery_days=2500,
    ),
}


# ヘッジ資産の特性定義
HEDGE_ASSETS: Dict[str, Dict[str, Any]] = {
    "TLT": {
        "name": "Long-term Treasury",
        "typical_crisis_return": 0.15,
        "annual_cost": -0.02,  # 機会コスト（低リターン）
        "effective_against": ["equity_crash", "deflation"],
    },
    "GLD": {
        "name": "Gold",
        "typical_crisis_return": 0.10,
        "annual_cost": -0.005,  # 保管コスト
        "effective_against": ["inflation", "currency_crisis", "tail_risk"],
    },
    "VXX": {
        "name": "VIX Futures",
        "typical_crisis_return": 0.50,
        "annual_cost": -0.40,  # ロールコスト高い
        "effective_against": ["equity_crash", "flash_crash"],
    },
    "TIP": {
        "name": "TIPS",
        "typical_crisis_return": 0.05,
        "annual_cost": 0.0,
        "effective_against": ["inflation", "stagflation"],
    },
    "UUP": {
        "name": "US Dollar",
        "typical_crisis_return": 0.08,
        "annual_cost": 0.0,
        "effective_against": ["em_crisis", "global_risk_off"],
    },
}


# =============================================================================
# StressTester クラス
# =============================================================================

class StressTester:
    """
    ストレステスト実行クラス

    歴史的なストレスシナリオに対するポートフォリオの耐性をテストする。

    Usage:
        tester = StressTester()

        # 単一シナリオテスト
        result = tester.run_stress_test(weights, HISTORICAL_SCENARIOS["GFC_2008"])

        # 全シナリオテスト
        results_df = tester.run_all_stress_tests(weights)

        # 最悪ケース損失
        worst_loss = tester.compute_worst_case_loss(weights)

        # ヘッジ提案
        hedges = tester.suggest_hedges(weights)
    """

    DEFAULT_SURVIVAL_THRESHOLD = -0.30  # 30%損失が生存閾値

    def __init__(
        self,
        scenarios: Optional[List[StressScenario]] = None,
        survival_threshold: float = -0.30,
        default_shock: float = -0.10,
    ) -> None:
        """
        初期化

        Args:
            scenarios: 使用するシナリオのリスト（Noneで全歴史的シナリオ）
            survival_threshold: 生存判定の損失閾値
            default_shock: 未定義資産のデフォルトショック
        """
        if scenarios is None:
            self.scenarios = list(HISTORICAL_SCENARIOS.values())
        else:
            self.scenarios = scenarios

        self.survival_threshold = survival_threshold
        self.default_shock = default_shock

        logger.info(
            f"StressTester initialized with {len(self.scenarios)} scenarios"
        )

    def run_stress_test(
        self,
        weights: Dict[str, float],
        scenario: StressScenario,
    ) -> StressTestResult:
        """
        単一シナリオのストレステストを実行

        Args:
            weights: ポートフォリオウェイト
            scenario: ストレスシナリオ

        Returns:
            StressTestResult
        """
        portfolio_loss = 0.0
        asset_impacts = {}

        for asset, weight in weights.items():
            if weight == 0:
                continue

            # 資産のショックを取得（未定義はデフォルトショック）
            shock = scenario.get_shock(asset, self.default_shock)

            # 相関乗数を適用（リスク資産のみ）
            if shock < 0:
                # 下落資産は相関乗数で損失拡大
                adjusted_shock = shock * scenario.correlation_multiplier
                # ただし-100%は超えない
                adjusted_shock = max(adjusted_shock, -1.0)
            else:
                adjusted_shock = shock

            # ポートフォリオへの影響
            impact = weight * adjusted_shock
            asset_impacts[asset] = impact
            portfolio_loss += impact

        # 生存判定
        survives = portfolio_loss >= self.survival_threshold

        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_loss=portfolio_loss,
            asset_impacts=asset_impacts,
            survives=survives,
            weights=weights,
            scenario=scenario,
        )

    def run_all_stress_tests(
        self,
        weights: Dict[str, float],
    ) -> pd.DataFrame:
        """
        全シナリオのストレステストを実行

        Args:
            weights: ポートフォリオウェイト

        Returns:
            シナリオ別結果のDataFrame
        """
        results = []

        for scenario in self.scenarios:
            result = self.run_stress_test(weights, scenario)
            results.append({
                "scenario": result.scenario_name,
                "portfolio_loss": result.portfolio_loss,
                "survives": result.survives,
                "description": scenario.description,
                "duration_days": scenario.duration_days,
                "recovery_days": scenario.recovery_days,
            })

        df = pd.DataFrame(results)
        df = df.sort_values("portfolio_loss", ascending=True)

        return df

    def compute_worst_case_loss(
        self,
        weights: Dict[str, float],
    ) -> float:
        """
        最悪ケースの損失を計算

        Args:
            weights: ポートフォリオウェイト

        Returns:
            最悪シナリオでの損失率
        """
        worst_loss = 0.0

        for scenario in self.scenarios:
            result = self.run_stress_test(weights, scenario)
            if result.portfolio_loss < worst_loss:
                worst_loss = result.portfolio_loss

        return worst_loss

    def get_worst_scenario(
        self,
        weights: Dict[str, float],
    ) -> Tuple[str, float]:
        """
        最悪シナリオとその損失を取得

        Args:
            weights: ポートフォリオウェイト

        Returns:
            (シナリオ名, 損失率)
        """
        worst_scenario = None
        worst_loss = 0.0

        for scenario in self.scenarios:
            result = self.run_stress_test(weights, scenario)
            if result.portfolio_loss < worst_loss:
                worst_loss = result.portfolio_loss
                worst_scenario = scenario.name

        return worst_scenario, worst_loss

    def suggest_hedges(
        self,
        weights: Dict[str, float],
        max_hedge_cost: float = 0.05,
        target_protection: float = 0.10,
    ) -> List[HedgeRecommendation]:
        """
        最悪シナリオに基づくヘッジ提案

        Args:
            weights: 現在のポートフォリオウェイト
            max_hedge_cost: 許容する最大ヘッジコスト（年率）
            target_protection: 目標保護効果

        Returns:
            HedgeRecommendationのリスト
        """
        # 最悪シナリオを特定
        worst_scenario_name, worst_loss = self.get_worst_scenario(weights)

        if worst_scenario_name is None:
            return []

        worst_scenario = HISTORICAL_SCENARIOS.get(worst_scenario_name)
        if worst_scenario is None:
            return []

        recommendations = []

        # 各ヘッジ資産を評価
        for hedge_asset, hedge_info in HEDGE_ASSETS.items():
            # 既にポートフォリオに含まれている場合はスキップ
            if weights.get(hedge_asset, 0) > 0.1:
                continue

            # シナリオでのヘッジ資産のリターン
            hedge_return = worst_scenario.get_shock(hedge_asset, 0.0)

            # ポジティブリターンの場合のみヘッジとして有効
            if hedge_return <= 0:
                continue

            # コストチェック
            annual_cost = hedge_info.get("annual_cost", 0.0)
            if abs(annual_cost) > max_hedge_cost:
                continue

            # 推奨ウェイト計算（目標保護 / ヘッジリターン）
            recommended_weight = min(
                target_protection / hedge_return,
                0.20,  # 最大20%
            )

            # 期待保護効果
            expected_protection = recommended_weight * hedge_return

            # 推奨理由
            effective_against = hedge_info.get("effective_against", [])
            rationale = (
                f"{hedge_info['name']}: {worst_scenario_name}シナリオで"
                f"+{hedge_return:.0%}のリターン期待。"
                f"有効シナリオ: {', '.join(effective_against)}"
            )

            recommendations.append(HedgeRecommendation(
                hedge_asset=hedge_asset,
                recommended_weight=recommended_weight,
                expected_protection=expected_protection,
                cost_estimate=abs(annual_cost),
                rationale=rationale,
            ))

        # 期待保護効果でソート
        recommendations.sort(key=lambda x: x.expected_protection, reverse=True)

        return recommendations

    def create_hedged_portfolio(
        self,
        weights: Dict[str, float],
        hedge_recommendations: List[HedgeRecommendation],
        max_hedge_allocation: float = 0.20,
    ) -> Dict[str, float]:
        """
        ヘッジを適用したポートフォリオを作成

        Args:
            weights: 元のウェイト
            hedge_recommendations: ヘッジ提案
            max_hedge_allocation: ヘッジ資産への最大配分

        Returns:
            ヘッジ適用後のウェイト
        """
        hedged_weights = weights.copy()

        # ヘッジ資産の配分
        total_hedge = 0.0
        for rec in hedge_recommendations:
            if total_hedge + rec.recommended_weight > max_hedge_allocation:
                allocation = max_hedge_allocation - total_hedge
            else:
                allocation = rec.recommended_weight

            if allocation > 0:
                hedged_weights[rec.hedge_asset] = allocation
                total_hedge += allocation

            if total_hedge >= max_hedge_allocation:
                break

        # 正規化
        total = sum(hedged_weights.values())
        if total > 0:
            hedged_weights = {k: v / total for k, v in hedged_weights.items()}

        return hedged_weights

    def compare_portfolios(
        self,
        original_weights: Dict[str, float],
        hedged_weights: Dict[str, float],
    ) -> pd.DataFrame:
        """
        オリジナルとヘッジ後のポートフォリオを比較

        Args:
            original_weights: 元のウェイト
            hedged_weights: ヘッジ後のウェイト

        Returns:
            比較結果のDataFrame
        """
        results = []

        for scenario in self.scenarios:
            orig_result = self.run_stress_test(original_weights, scenario)
            hedged_result = self.run_stress_test(hedged_weights, scenario)

            improvement = hedged_result.portfolio_loss - orig_result.portfolio_loss

            results.append({
                "scenario": scenario.name,
                "original_loss": orig_result.portfolio_loss,
                "hedged_loss": hedged_result.portfolio_loss,
                "improvement": improvement,
                "orig_survives": orig_result.survives,
                "hedged_survives": hedged_result.survives,
            })

        return pd.DataFrame(results)

    def get_scenario_summary(self) -> pd.DataFrame:
        """利用可能なシナリオのサマリーを取得"""
        data = []
        for scenario in self.scenarios:
            data.append({
                "name": scenario.name,
                "description": scenario.description,
                "duration_days": scenario.duration_days,
                "recovery_days": scenario.recovery_days,
                "correlation_multiplier": scenario.correlation_multiplier,
                "n_assets": len(scenario.shocks),
            })
        return pd.DataFrame(data)


# =============================================================================
# 便利関数
# =============================================================================

def run_stress_test(
    weights: Dict[str, float],
    scenario_name: str,
) -> StressTestResult:
    """
    ストレステストを実行（便利関数）

    Args:
        weights: ポートフォリオウェイト
        scenario_name: シナリオ名

    Returns:
        StressTestResult
    """
    scenario = HISTORICAL_SCENARIOS.get(scenario_name)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    tester = StressTester()
    return tester.run_stress_test(weights, scenario)


def get_worst_case_loss(weights: Dict[str, float]) -> float:
    """
    最悪ケース損失を計算（便利関数）

    Args:
        weights: ポートフォリオウェイト

    Returns:
        最悪シナリオでの損失率
    """
    tester = StressTester()
    return tester.compute_worst_case_loss(weights)


def list_scenarios() -> List[str]:
    """利用可能なシナリオ名のリストを取得"""
    return list(HISTORICAL_SCENARIOS.keys())


def create_stress_tester(
    scenarios: Optional[List[str]] = None,
    survival_threshold: float = -0.30,
) -> StressTester:
    """
    StressTesterを作成（ファクトリ関数）

    Args:
        scenarios: 使用するシナリオ名のリスト
        survival_threshold: 生存判定の損失閾値

    Returns:
        StressTester
    """
    if scenarios is not None:
        scenario_list = [
            HISTORICAL_SCENARIOS[name]
            for name in scenarios
            if name in HISTORICAL_SCENARIOS
        ]
    else:
        scenario_list = None

    return StressTester(
        scenarios=scenario_list,
        survival_threshold=survival_threshold,
    )
