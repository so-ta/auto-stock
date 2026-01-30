"""
Meta Learner Module - Meta層統合

Scorer, Weighter, EntropyControllerを統合し、
ゲート合格戦略から最終的な重み配分を決定する。

Phase 2: 階層的アンサンブル統合
- HierarchicalEnsemble: 3レイヤー構成のシグナル統合
- AdaptiveLookback: レジーム適応的ルックバック
- BayesianOptimizer: パラメータ自動最適化

処理フロー:
【従来モード】
1. ゲートチェック結果を受け取り、合格戦略のみ抽出
2. Scorerで各戦略のスコアを計算
3. Weighterでスコアから重みを計算
4. EntropyControllerで多様性を確保

【階層的アンサンブルモード】
1. レジーム検出（RegimeDetector / AdaptiveLookback）
2. 適応的ルックバック選択（AdaptiveLookback）
3. 階層的アンサンブル（HierarchicalEnsemble）
   - Trend Layer → スタッキング
   - Reversion Layer → スタッキング
   - Macro Layer → スタッキング
4. レジーム適応レイヤー統合
5. 最終スコア → アセット配分

設計根拠:
- 要求.md §7: Strategyの採用と重み
- 統合設計: 各コンポーネントの責務分離と連携
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from .entropy_controller import EntropyConfig, EntropyControlResult, EntropyController
from .scorer import ScorerConfig, StrategyMetricsInput, StrategyScorer, StrategyScoreResult
from .weighter import StrategyWeighter, WeighterConfig, WeightingResult

if TYPE_CHECKING:
    from .adaptive_lookback import AdaptiveLookback, AdaptiveLookbackResult
    from .bayesian_optimizer import BayesianOptimizer, OptimizationResult
    from .hierarchical_ensemble import HierarchicalEnsemble, HierarchicalEnsembleResult
    from .param_consistency import ParameterConsistencyChecker, ConsistencyResult

logger = logging.getLogger(__name__)


@dataclass
class StrategyWeight:
    """戦略重み情報

    Attributes:
        strategy_id: 戦略ID
        asset_id: アセットID
        weight: 最終重み
        score: スコア
        is_adopted: 採用されたか（ゲート合格かつ重み > 0）
    """

    strategy_id: str
    asset_id: str
    weight: float
    score: float
    is_adopted: bool = True

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "strategy_id": self.strategy_id,
            "asset_id": self.asset_id,
            "weight": self.weight,
            "score": self.score,
            "is_adopted": self.is_adopted,
        }


@dataclass
class MetaLearnerResult:
    """Meta学習結果

    Attributes:
        asset_id: アセットID
        strategy_weights: 戦略重みリスト
        total_adopted: 採用された戦略数
        total_evaluated: 評価された戦略数（ゲート合格数）
        scoring_results: スコアリング結果
        weighting_result: 重み計算結果
        entropy_result: エントロピー制御結果
        computed_at: 計算日時
        metadata: 追加メタデータ
    """

    asset_id: str
    strategy_weights: list[StrategyWeight] = field(default_factory=list)
    total_adopted: int = 0
    total_evaluated: int = 0
    scoring_results: list[StrategyScoreResult] = field(default_factory=list)
    weighting_result: WeightingResult | None = None
    entropy_result: EntropyControlResult | None = None
    computed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "asset_id": self.asset_id,
            "strategy_weights": [w.to_dict() for w in self.strategy_weights],
            "total_adopted": self.total_adopted,
            "total_evaluated": self.total_evaluated,
            "scoring_results": [r.to_dict() for r in self.scoring_results],
            "weighting_result": self.weighting_result.to_dict() if self.weighting_result else None,
            "entropy_result": self.entropy_result.to_dict() if self.entropy_result else None,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
            "metadata": self.metadata,
        }

    def get_weight_dict(self) -> dict[str, float]:
        """strategy_id -> weight の辞書を取得"""
        return {w.strategy_id: w.weight for w in self.strategy_weights}

    def get_adopted_strategies(self) -> list[str]:
        """採用された戦略IDリストを取得"""
        return [w.strategy_id for w in self.strategy_weights if w.is_adopted]


@dataclass
class MetaLearnerConfig:
    """Meta Learner設定

    Attributes:
        scorer_config: Scorer設定
        weighter_config: Weighter設定
        entropy_config: EntropyController設定
        min_strategies_required: 最低必要戦略数
        fallback_to_equal_weight: 戦略不足時に均等配分するか
        use_hierarchical_ensemble: 階層的アンサンブルを使用するか
        stacking_model_type: スタッキングモデルタイプ (ridge | xgboost | simple_avg)
        use_adaptive_lookback: 適応的ルックバックを使用するか
        enable_bayesian_optimization: ベイズ最適化を有効にするか
        enable_param_consistency: パラメータ整合性チェックを有効にするか
        auto_adjust_params: 不整合パラメータを自動調整するか
    """

    scorer_config: ScorerConfig = field(default_factory=ScorerConfig)
    weighter_config: WeighterConfig = field(default_factory=WeighterConfig)
    entropy_config: EntropyConfig = field(default_factory=EntropyConfig)
    min_strategies_required: int = 1
    fallback_to_equal_weight: bool = True

    # 階層的アンサンブル設定
    use_hierarchical_ensemble: bool = False
    stacking_model_type: str = "ridge"
    use_adaptive_lookback: bool = True
    enable_bayesian_optimization: bool = False

    # パラメータ整合性設定 (Phase 4)
    enable_param_consistency: bool = True
    auto_adjust_params: bool = True


class MetaLearner:
    """Meta層統合クラス

    Scorer, Weighter, EntropyControllerを統合し、
    ゲート合格戦略から最終的な重み配分を決定する。

    Usage:
        learner = MetaLearner()

        # 戦略メトリクス（ゲート合格済み）
        metrics_list = [
            StrategyMetricsInput(
                strategy_id="momentum",
                asset_id="AAPL",
                sharpe_ratio=1.5,
                max_drawdown_pct=15.0,
            ),
            ...
        ]

        result = learner.compute_weights(metrics_list, asset_id="AAPL")
        print(f"Adopted: {result.get_adopted_strategies()}")
        print(f"Weights: {result.get_weight_dict()}")
    """

    def __init__(self, config: MetaLearnerConfig | None = None) -> None:
        """初期化

        Args:
            config: Meta Learner設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or MetaLearnerConfig()
        self.scorer = StrategyScorer(self.config.scorer_config)
        self.weighter = StrategyWeighter(self.config.weighter_config)
        self.entropy_controller = EntropyController(self.config.entropy_config)

        # 階層的アンサンブル関連（遅延初期化）
        self._hierarchical_ensemble: Optional["HierarchicalEnsemble"] = None
        self._adaptive_lookback: Optional["AdaptiveLookback"] = None
        self._bayesian_optimizer: Optional["BayesianOptimizer"] = None
        self._is_fitted: bool = False

        # パラメータ整合性チェッカー（Phase 4）
        self._param_consistency_checker: Optional["ParameterConsistencyChecker"] = None
        if self.config.enable_param_consistency:
            self._init_param_consistency_checker()

        # 階層的アンサンブルモードの場合は初期化
        if self.config.use_hierarchical_ensemble:
            self._init_hierarchical_ensemble()

    def _init_param_consistency_checker(self) -> None:
        """パラメータ整合性チェッカーを初期化"""
        try:
            from .param_consistency import ParameterConsistencyChecker

            self._param_consistency_checker = ParameterConsistencyChecker()
            logger.info("Parameter consistency checker initialized")
        except ImportError as e:
            logger.warning(f"ParameterConsistencyChecker not available: {e}")
            self._param_consistency_checker = None

    def check_param_consistency(self) -> "ConsistencyResult | None":
        """現在のパラメータ設定の整合性をチェック

        Returns:
            ConsistencyResult or None if checker not available
        """
        if self._param_consistency_checker is None:
            return None

        # 現在のパラメータを収集
        params = {
            "scorer": {
                "penalty_turnover": self.config.scorer_config.penalty_turnover,
                "penalty_mdd": self.config.scorer_config.penalty_mdd,
                "penalty_instability": self.config.scorer_config.penalty_instability,
                "return_bonus_scale": self.config.scorer_config.return_bonus_scale,
                "alpha_bonus_scale": self.config.scorer_config.alpha_bonus_scale,
            },
            "weighter": {
                "beta": self.config.weighter_config.beta,
                "w_strategy_max": self.config.weighter_config.w_strategy_max,
                "score_threshold": self.config.weighter_config.score_threshold,
            },
            "entropy": {
                "entropy_min": self.config.entropy_config.entropy_min,
                "entropy_max": self.config.entropy_config.entropy_max,
            },
        }

        # 整合性チェック
        result = self._param_consistency_checker.check(params)

        if not result.is_consistent:
            logger.warning(
                "Parameter consistency issues detected: %s",
                [str(issue) for issue in result.issues],
            )

            # 自動調整が有効な場合
            if self.config.auto_adjust_params and result.adjusted_params:
                self._apply_adjusted_params(result.adjusted_params)
                logger.info("Parameters auto-adjusted for consistency")

        return result

    def _apply_adjusted_params(self, adjusted_params: dict[str, Any]) -> None:
        """調整されたパラメータを適用

        Args:
            adjusted_params: 調整後のパラメータ辞書
        """
        if "weighter" in adjusted_params:
            weighter_params = adjusted_params["weighter"]
            if "beta" in weighter_params:
                self.config.weighter_config.beta = weighter_params["beta"]
            if "w_strategy_max" in weighter_params:
                self.config.weighter_config.w_strategy_max = weighter_params["w_strategy_max"]
            # Weighter を再初期化
            self.weighter = StrategyWeighter(self.config.weighter_config)

        if "scorer" in adjusted_params:
            scorer_params = adjusted_params["scorer"]
            # ScorerConfig は frozen=True なので新規作成
            new_scorer_config = ScorerConfig(
                penalty_turnover=scorer_params.get("penalty_turnover", self.config.scorer_config.penalty_turnover),
                penalty_mdd=scorer_params.get("penalty_mdd", self.config.scorer_config.penalty_mdd),
                penalty_instability=scorer_params.get("penalty_instability", self.config.scorer_config.penalty_instability),
                return_bonus_scale=scorer_params.get("return_bonus_scale", self.config.scorer_config.return_bonus_scale),
                alpha_bonus_scale=scorer_params.get("alpha_bonus_scale", self.config.scorer_config.alpha_bonus_scale),
            )
            self.config.scorer_config = new_scorer_config
            self.scorer = StrategyScorer(new_scorer_config)

    def compute_weights(
        self,
        metrics_list: list[StrategyMetricsInput],
        asset_id: str | None = None,
    ) -> MetaLearnerResult:
        """戦略重みを計算

        Args:
            metrics_list: ゲート合格済み戦略のメトリクスリスト
            asset_id: アセットID（省略時は最初のメトリクスから取得）

        Returns:
            Meta学習結果
        """
        if asset_id is None:
            asset_id = metrics_list[0].asset_id if metrics_list else "unknown"

        # パラメータ整合性チェック (Phase 4)
        if self.config.enable_param_consistency:
            consistency_result = self.check_param_consistency()
            if consistency_result and not consistency_result.is_consistent:
                logger.debug(
                    "Parameter consistency check: %d issues found",
                    len(consistency_result.issues),
                )

        result = MetaLearnerResult(
            asset_id=asset_id,
            total_evaluated=len(metrics_list),
            computed_at=datetime.now(),
        )

        # 空の場合
        if not metrics_list:
            logger.warning("No strategies provided for asset %s", asset_id)
            result.metadata["error"] = "No strategies provided"
            return result

        # 戦略数不足チェック
        if len(metrics_list) < self.config.min_strategies_required:
            logger.warning(
                "Insufficient strategies for %s: %d < %d",
                asset_id,
                len(metrics_list),
                self.config.min_strategies_required,
            )
            if self.config.fallback_to_equal_weight:
                return self._fallback_equal_weight(metrics_list, asset_id, result)
            result.metadata["error"] = "Insufficient strategies"
            return result

        # Step 1: スコアリング
        scoring_results = self.scorer.score_batch(metrics_list)
        result.scoring_results = scoring_results

        # Step 2: 重み計算
        weighting_result = self.weighter.calculate_weights(scoring_results)
        result.weighting_result = weighting_result

        # Step 3: エントロピー制御
        weights_dict = weighting_result.get_weight_dict()
        entropy_result = self.entropy_controller.control(weights_dict)
        result.entropy_result = entropy_result

        # 最終重みを構築
        final_weights = entropy_result.adjusted_weights
        strategy_weights = []

        for score_result in scoring_results:
            weight = final_weights.get(score_result.strategy_id, 0.0)
            is_adopted = weight > 0

            strategy_weight = StrategyWeight(
                strategy_id=score_result.strategy_id,
                asset_id=asset_id,
                weight=weight,
                score=score_result.final_score,
                is_adopted=is_adopted,
            )
            strategy_weights.append(strategy_weight)

        result.strategy_weights = strategy_weights
        result.total_adopted = sum(1 for w in strategy_weights if w.is_adopted)

        logger.info(
            "Meta learner computed weights for %s: %d/%d adopted, entropy=%.3f",
            asset_id,
            result.total_adopted,
            result.total_evaluated,
            entropy_result.adjusted_entropy,
        )

        return result

    def compute_weights_from_evaluations(
        self,
        evaluations: list[dict[str, Any]],
        asset_id: str,
    ) -> MetaLearnerResult:
        """評価結果辞書から重みを計算

        Args:
            evaluations: 戦略評価結果のリスト（辞書形式）
            asset_id: アセットID

        Returns:
            Meta学習結果
        """
        metrics_list = []

        for eval_dict in evaluations:
            # ゲート不合格はスキップ
            if not eval_dict.get("is_adopted", True):
                continue

            metrics_data = eval_dict.get("metrics", {})
            metrics = StrategyMetricsInput(
                strategy_id=eval_dict.get("strategy_id", "unknown"),
                asset_id=asset_id,
                sharpe_ratio=metrics_data.get("sharpe_ratio", 0.0),
                max_drawdown_pct=metrics_data.get("max_drawdown_pct", 0.0),
                turnover=metrics_data.get("turnover", 0.0),
                period_returns=metrics_data.get("period_returns", []),
            )
            metrics_list.append(metrics)

        return self.compute_weights(metrics_list, asset_id)

    def _fallback_equal_weight(
        self,
        metrics_list: list[StrategyMetricsInput],
        asset_id: str,
        result: MetaLearnerResult,
    ) -> MetaLearnerResult:
        """均等配分にフォールバック

        Args:
            metrics_list: メトリクスリスト
            asset_id: アセットID
            result: 結果オブジェクト（更新される）

        Returns:
            更新された結果
        """
        n = len(metrics_list)
        equal_weight = 1.0 / n if n > 0 else 0.0

        strategy_weights = []
        for metrics in metrics_list:
            strategy_weight = StrategyWeight(
                strategy_id=metrics.strategy_id,
                asset_id=asset_id,
                weight=equal_weight,
                score=0.0,  # スコア計算スキップ
                is_adopted=True,
            )
            strategy_weights.append(strategy_weight)

        result.strategy_weights = strategy_weights
        result.total_adopted = n
        result.metadata["fallback"] = "equal_weight"
        result.metadata["reason"] = "insufficient_strategies"

        logger.info(
            "Fallback to equal weight for %s: %d strategies with weight %.3f",
            asset_id,
            n,
            equal_weight,
        )

        return result

    # =========================================================================
    # 階層的アンサンブル統合メソッド
    # =========================================================================

    def _init_hierarchical_ensemble(self) -> None:
        """階層的アンサンブル関連コンポーネントを初期化"""
        from .adaptive_lookback import AdaptiveLookback
        from .hierarchical_ensemble import create_hierarchical_ensemble_with_signals

        # 階層的アンサンブル
        self._hierarchical_ensemble = create_hierarchical_ensemble_with_signals(
            stacking_model_type=self.config.stacking_model_type,
        )

        # 適応的ルックバック
        if self.config.use_adaptive_lookback:
            self._adaptive_lookback = AdaptiveLookback()

        # ベイズ最適化（オプション）
        if self.config.enable_bayesian_optimization:
            from .bayesian_optimizer import BayesianOptimizer, OptimizerConfig

            self._bayesian_optimizer = BayesianOptimizer(
                config=OptimizerConfig(n_calls=50, n_random_starts=15)
            )

        logger.info(
            "Hierarchical ensemble initialized: stacking=%s, adaptive_lookback=%s, bayesian=%s",
            self.config.stacking_model_type,
            self.config.use_adaptive_lookback,
            self.config.enable_bayesian_optimization,
        )

    @property
    def hierarchical_ensemble(self) -> Optional["HierarchicalEnsemble"]:
        """階層的アンサンブルインスタンスを取得"""
        return self._hierarchical_ensemble

    @property
    def adaptive_lookback(self) -> Optional["AdaptiveLookback"]:
        """適応的ルックバックインスタンスを取得"""
        return self._adaptive_lookback

    @property
    def bayesian_optimizer(self) -> Optional["BayesianOptimizer"]:
        """ベイズ最適化インスタンスを取得"""
        return self._bayesian_optimizer

    def fit_hierarchical(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        additional_data: Optional[dict[str, Any]] = None,
    ) -> "MetaLearner":
        """階層的アンサンブルモデルを学習

        Args:
            prices: 価格DataFrame (OHLCV)
            returns: ターゲットリターン
            additional_data: 追加データ（マクロ指標など）

        Returns:
            self
        """
        if self._hierarchical_ensemble is None:
            self._init_hierarchical_ensemble()

        if self._hierarchical_ensemble is None:
            raise RuntimeError("Failed to initialize hierarchical ensemble")

        self._hierarchical_ensemble.fit(prices, returns, additional_data)
        self._is_fitted = True

        logger.info("Hierarchical ensemble fitted successfully")
        return self

    def predict_hierarchical(
        self,
        prices: pd.DataFrame,
        regime: Optional[str] = None,
        additional_data: Optional[dict[str, Any]] = None,
    ) -> "HierarchicalEnsembleResult":
        """階層的アンサンブルで予測

        Args:
            prices: 価格DataFrame (OHLCV)
            regime: レジーム名（省略時は自動検出）
            additional_data: 追加データ

        Returns:
            HierarchicalEnsembleResult
        """
        if self._hierarchical_ensemble is None:
            raise RuntimeError("Hierarchical ensemble not initialized")

        # レジーム自動検出
        if regime is None:
            if self._adaptive_lookback is not None:
                regime = self._adaptive_lookback.detect_regime(prices)
            else:
                regime = self._hierarchical_ensemble.detect_regime(prices, additional_data)

        logger.debug(f"Using regime: {regime}")

        return self._hierarchical_ensemble.predict(prices, regime, additional_data)

    def learn(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        universe: Optional[list[str]] = None,
        additional_data: Optional[dict[str, Any]] = None,
    ) -> MetaLearnerResult:
        """統合学習メソッド

        階層的アンサンブルモードの場合:
        1. レジーム検出
        2. 適応的ルックバック選択
        3. 階層的アンサンブル予測
        4. スコアから重みを計算

        従来モードの場合:
        compute_weights を使用

        Args:
            prices: 価格DataFrame
            returns: リターン系列
            universe: アセットユニバース
            additional_data: 追加データ

        Returns:
            MetaLearnerResult
        """
        if not self.config.use_hierarchical_ensemble:
            # 従来モード（compute_weights用のデータが必要）
            logger.info("Using legacy mode (compute_weights)")
            return MetaLearnerResult(
                asset_id="portfolio",
                computed_at=datetime.now(),
                metadata={"mode": "legacy", "note": "Use compute_weights with metrics"},
            )

        # 階層的アンサンブルモード
        logger.info("Using hierarchical ensemble mode")

        # Step 1: 学習（未学習の場合）
        if not self._is_fitted:
            self.fit_hierarchical(prices, returns, additional_data)

        # Step 2: レジーム検出
        if self._adaptive_lookback is not None:
            regime = self._adaptive_lookback.detect_regime(prices)
        elif self._hierarchical_ensemble is not None:
            regime = self._hierarchical_ensemble.detect_regime(prices, additional_data)
        else:
            regime = "default"

        # Step 3: 適応的ルックバック（オプション）
        lookback_result: Optional["AdaptiveLookbackResult"] = None
        if self._adaptive_lookback is not None:
            # シンプルなモメンタムシグナル関数
            def momentum_signal(data: pd.DataFrame, lookback: int) -> pd.Series:
                return data["close"].pct_change(lookback)

            lookback_result = self._adaptive_lookback.compute_multi_period_score(
                prices, momentum_signal, regime
            )

        # Step 4: 階層的アンサンブル予測
        ensemble_result = self.predict_hierarchical(prices, regime, additional_data)

        # Step 5: スコアから戦略重みを構築
        final_score = ensemble_result.final_score
        strategy_weights = []

        # 各レイヤーの結果から戦略重みを作成
        for layer_name, layer_result in [
            ("trend", ensemble_result.trend_result),
            ("reversion", ensemble_result.reversion_result),
            ("macro", ensemble_result.macro_result),
        ]:
            if layer_result.is_valid:
                layer_weight = getattr(ensemble_result.layer_weights, layer_name)
                for signal_name, signal_weight in layer_result.weights_used.items():
                    strategy_weights.append(
                        StrategyWeight(
                            strategy_id=f"{layer_name}_{signal_name}",
                            asset_id="portfolio",
                            weight=signal_weight * layer_weight,
                            score=float(layer_result.score.mean()),
                            is_adopted=signal_weight > 0,
                        )
                    )

        result = MetaLearnerResult(
            asset_id="portfolio",
            strategy_weights=strategy_weights,
            total_adopted=sum(1 for w in strategy_weights if w.is_adopted),
            total_evaluated=len(strategy_weights),
            computed_at=datetime.now(),
            metadata={
                "mode": "hierarchical_ensemble",
                "regime": regime,
                "final_score_mean": float(final_score.mean()),
                "final_score_std": float(final_score.std()),
                "layer_weights": ensemble_result.layer_weights.to_dict(),
                "lookback_periods": (
                    lookback_result.lookback_periods if lookback_result else None
                ),
            },
        )

        logger.info(
            "Hierarchical learning completed: regime=%s, adopted=%d, score=%.3f",
            regime,
            result.total_adopted,
            float(final_score.mean()),
        )

        return result

    def optimize_params(
        self,
        train_data: pd.DataFrame,
        universe: list[str],
        n_calls: int = 50,
    ) -> "OptimizationResult":
        """ベイズ最適化でパラメータを最適化

        Args:
            train_data: 学習データ
            universe: 銘柄リスト
            n_calls: 最適化試行回数

        Returns:
            OptimizationResult
        """
        if self._bayesian_optimizer is None:
            from .bayesian_optimizer import BayesianOptimizer, OptimizerConfig

            self._bayesian_optimizer = BayesianOptimizer(
                config=OptimizerConfig(n_calls=n_calls, n_random_starts=min(15, n_calls // 3))
            )

        logger.info(f"Starting parameter optimization with {n_calls} calls")

        result = self._bayesian_optimizer.optimize(
            train_data=train_data,
            universe=universe,
        )

        # 最適パラメータをconfigに反映
        if result.best_params:
            self._apply_optimized_params(result.best_params)

        logger.info(
            "Optimization completed: best_sharpe=%.3f, params=%s",
            result.best_score,
            result.best_params,
        )

        return result

    def _apply_optimized_params(self, params: dict[str, Any]) -> None:
        """最適化されたパラメータを設定に反映

        Args:
            params: 最適化されたパラメータ辞書
        """
        # Weighter設定
        if "beta" in params:
            self.config.weighter_config.beta = params["beta"]
            self.weighter = StrategyWeighter(self.config.weighter_config)

        # 将来の拡張: その他のパラメータも反映
        logger.info(f"Applied optimized params: {params}")


def create_meta_learner_from_settings() -> MetaLearner:
    """グローバル設定からMetaLearnerを生成

    config/default.yaml の meta_learner セクションと
    strategy_weighting セクションから設定を読み込む。

    Returns:
        設定済みのMetaLearner
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        sw = settings.strategy_weighting

        # meta_learner設定があれば読み込み
        ml_config = getattr(settings, "meta_learner", None)

        # return_maximization設定（Phase 4）
        rm_config = getattr(settings, "return_maximization", None)
        scoring_config = getattr(rm_config, "scoring", None) if rm_config else None
        param_consistency_config = getattr(rm_config, "param_consistency", None) if rm_config else None

        # ScorerConfig にリターンボーナス設定を追加
        scorer_config = ScorerConfig(
            penalty_turnover=sw.penalty_turnover,
            penalty_mdd=sw.penalty_mdd,
            penalty_instability=sw.penalty_instability,
            return_bonus_scale=getattr(scoring_config, "return_bonus_scale", 0.3) if scoring_config else 0.3,
            alpha_bonus_scale=getattr(scoring_config, "alpha_bonus_scale", 0.5) if scoring_config else 0.5,
            benchmark_ticker=getattr(scoring_config, "benchmark_ticker", "SPY") if scoring_config else "SPY",
        )

        config = MetaLearnerConfig(
            scorer_config=scorer_config,
            weighter_config=WeighterConfig(
                beta=sw.beta,
                w_strategy_max=sw.w_strategy_max,
            ),
            entropy_config=EntropyConfig(
                entropy_min=sw.entropy_min,
            ),
            # 階層的アンサンブル設定
            use_hierarchical_ensemble=getattr(ml_config, "use_hierarchical_ensemble", False) if ml_config else False,
            stacking_model_type=getattr(ml_config, "stacking_model_type", "ridge") if ml_config else "ridge",
            use_adaptive_lookback=getattr(ml_config, "use_adaptive_lookback", True) if ml_config else True,
            enable_bayesian_optimization=getattr(ml_config, "enable_bayesian_optimization", False) if ml_config else False,
            # パラメータ整合性設定（Phase 4）
            enable_param_consistency=getattr(param_consistency_config, "auto_adjust", True) if param_consistency_config else True,
            auto_adjust_params=getattr(param_consistency_config, "auto_adjust", True) if param_consistency_config else True,
        )
        return MetaLearner(config)
    except ImportError:
        logger.warning("Settings not available, using default MetaLearnerConfig")
        return MetaLearner()


def create_hierarchical_meta_learner(
    stacking_model_type: str = "ridge",
    use_adaptive_lookback: bool = True,
    enable_bayesian_optimization: bool = False,
) -> MetaLearner:
    """階層的アンサンブルモードのMetaLearnerを生成

    Args:
        stacking_model_type: スタッキングモデルタイプ (ridge | xgboost | simple_avg)
        use_adaptive_lookback: 適応的ルックバックを使用するか
        enable_bayesian_optimization: ベイズ最適化を有効にするか

    Returns:
        階層的アンサンブルモードのMetaLearner
    """
    config = MetaLearnerConfig(
        use_hierarchical_ensemble=True,
        stacking_model_type=stacking_model_type,
        use_adaptive_lookback=use_adaptive_lookback,
        enable_bayesian_optimization=enable_bayesian_optimization,
    )
    return MetaLearner(config)
