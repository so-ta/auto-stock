"""
Return Predictor Module - リターン予測モデル

LightGBM/RandomForestによる直接リターン予測を提供する。
価格データからテクニカル特徴量を生成し、将来リターンを予測。

設計根拠:
- 要求.md §7: Meta層
- 機械学習によるリターン予測

主な機能:
- テクニカル特徴量の自動生成
- 将来リターンの予測
- 特徴量重要度の分析

特徴量:
- リターン系: ret_{lb}, vol_{lb}
- テクニカル: rsi, macd, bb_position
- 相対強弱: rel_str (vs SPY)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# オプショナル依存のチェック
_HAS_LIGHTGBM = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    logger.info("lightgbm not available, ReturnPredictor will use RandomForest")


@dataclass
class PredictorConfig:
    """予測モデル設定

    Attributes:
        model_type: モデルタイプ（'lightgbm' or 'random_forest'）
        prediction_horizon: 予測期間（日数）
        lookback_features: 特徴量生成用のルックバック期間リスト
        rsi_period: RSI計算期間
        macd_fast: MACD短期EMA期間
        macd_slow: MACD長期EMA期間
        macd_signal: MACDシグナル期間
        bb_period: ボリンジャーバンド期間
        benchmark: 相対強弱計算用のベンチマーク
    """
    model_type: str = "lightgbm"
    prediction_horizon: int = 20
    lookback_features: Tuple[int, ...] = (5, 10, 20, 60)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    benchmark: str = "SPY"


@dataclass
class PredictionResult:
    """予測結果

    Attributes:
        prediction: 予測リターン
        confidence: 予測信頼度（あれば）
        feature_values: 使用した特徴量の値
        timestamp: 予測時刻
    """
    prediction: float
    confidence: Optional[float] = None
    feature_values: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "feature_values": self.feature_values,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class ReturnPredictor:
    """リターン予測クラス

    LightGBM/RandomForestによる直接リターン予測。
    価格データからテクニカル特徴量を生成し、将来リターンを予測する。

    Usage:
        predictor = ReturnPredictor(
            model_type="lightgbm",
            prediction_horizon=20,
            lookback_features=[5, 10, 20, 60],
        )

        # 学習
        predictor.fit(prices_df, target_asset="AAPL")

        # 予測
        prediction = predictor.predict(prices_df)
        print(f"Predicted return: {prediction:.4f}")

    Attributes:
        config: 予測モデル設定
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        prediction_horizon: int = 20,
        lookback_features: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        """初期化

        Args:
            model_type: モデルタイプ（'lightgbm' or 'random_forest'）
            prediction_horizon: 予測期間（日数）
            lookback_features: 特徴量生成用のルックバック期間リスト
            **kwargs: その他のConfig設定
        """
        if lookback_features is None:
            lookback_features = [5, 10, 20, 60]

        self.config = PredictorConfig(
            model_type=model_type,
            prediction_horizon=prediction_horizon,
            lookback_features=tuple(lookback_features),
            **kwargs,
        )

        self._model: Any = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._target_asset: str = ""
        self._is_fitted = False
        self._use_fallback = not _HAS_LIGHTGBM and model_type == "lightgbm"

        if self._use_fallback:
            logger.warning(
                "LightGBM not available, using RandomForest fallback"
            )

    def _compute_rsi(
        self,
        price: pd.Series,
        period: Optional[int] = None,
    ) -> pd.Series:
        """RSI（相対力指数）を計算

        Args:
            price: 価格シリーズ
            period: 計算期間

        Returns:
            RSI値（0-100）
        """
        period = period or self.config.rsi_period
        delta = price.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _compute_macd(
        self,
        price: pd.Series,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
    ) -> pd.Series:
        """MACD（移動平均収束拡散）を計算

        Args:
            price: 価格シリーズ
            fast: 短期EMA期間
            slow: 長期EMA期間
            signal: シグナル期間

        Returns:
            MACDヒストグラム（MACD - Signal）
        """
        fast = fast or self.config.macd_fast
        slow = slow or self.config.macd_slow
        signal = signal or self.config.macd_signal

        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        return macd_line - signal_line

    def _compute_bb_position(
        self,
        price: pd.Series,
        period: Optional[int] = None,
    ) -> pd.Series:
        """ボリンジャーバンド内のポジションを計算

        Args:
            price: 価格シリーズ
            period: 計算期間

        Returns:
            -1（下バンド）から+1（上バンド）の位置
        """
        period = period or self.config.bb_period

        middle = price.rolling(window=period).mean()
        std = price.rolling(window=period).std()

        upper = middle + 2 * std
        lower = middle - 2 * std

        # -1（下バンド）から+1（上バンド）に正規化
        bb_position = 2 * (price - lower) / (upper - lower + 1e-10) - 1
        bb_position = bb_position.clip(-1, 1)

        return bb_position

    def create_features(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """特徴量を生成

        Args:
            prices: 価格データ（index=date, columns=tickers）
            volumes: ボリュームデータ（オプション）
            asset: 対象アセット（Noneの場合は全アセット）

        Returns:
            特徴量DataFrame
        """
        if asset is not None:
            if asset not in prices.columns:
                raise ValueError(f"Asset {asset} not found in prices")
            price = prices[asset]
        else:
            # 最初のカラムを使用
            price = prices.iloc[:, 0]
            asset = prices.columns[0]

        features: Dict[str, pd.Series] = {}

        # リターン系特徴量
        for lb in self.config.lookback_features:
            # 過去リターン
            features[f"ret_{lb}"] = price.pct_change(periods=lb)
            # 過去ボラティリティ
            features[f"vol_{lb}"] = (
                price.pct_change().rolling(window=lb).std() * np.sqrt(252)
            )

        # テクニカル特徴量
        features["rsi"] = self._compute_rsi(price) / 100.0 - 0.5  # -0.5 to 0.5
        features["macd"] = self._compute_macd(price)
        features["bb_position"] = self._compute_bb_position(price)

        # 相対強弱（ベンチマーク対比）
        benchmark = self.config.benchmark
        if benchmark in prices.columns and benchmark != asset:
            bench_price = prices[benchmark]
            rel_lb = self.config.lookback_features[-1]  # 最長ルックバックを使用

            asset_ret = price.pct_change(periods=rel_lb)
            bench_ret = bench_price.pct_change(periods=rel_lb)
            features["rel_str"] = asset_ret - bench_ret
        else:
            features["rel_str"] = pd.Series(0.0, index=price.index)

        # モメンタム（短期 vs 長期）
        if len(self.config.lookback_features) >= 2:
            short_lb = self.config.lookback_features[0]
            long_lb = self.config.lookback_features[-1]
            short_ret = price.pct_change(periods=short_lb)
            long_ret = price.pct_change(periods=long_lb)
            features["momentum_diff"] = short_ret - long_ret / (long_lb / short_lb)

        # ボリューム関連（あれば）
        if volumes is not None and asset in volumes.columns:
            vol = volumes[asset]
            vol_ma = vol.rolling(window=20).mean()
            features["vol_ratio"] = vol / (vol_ma + 1e-10) - 1

        # DataFrameに統合
        feature_df = pd.DataFrame(features)
        self._feature_names = list(feature_df.columns)

        return feature_df

    def create_target(
        self,
        prices: pd.DataFrame,
        asset: str,
    ) -> pd.Series:
        """ターゲット（将来リターン）を生成

        Args:
            prices: 価格データ
            asset: 対象アセット

        Returns:
            将来リターンのSeries
        """
        if asset not in prices.columns:
            raise ValueError(f"Asset {asset} not found in prices")

        price = prices[asset]
        horizon = self.config.prediction_horizon

        # 将来リターン（horizon日後のリターン）
        future_return = price.pct_change(periods=horizon).shift(-horizon)

        return future_return

    def fit(
        self,
        prices: pd.DataFrame,
        target_asset: str,
        volumes: Optional[pd.DataFrame] = None,
    ) -> "ReturnPredictor":
        """モデルを学習

        Args:
            prices: 価格データ
            target_asset: 予測対象のアセット
            volumes: ボリュームデータ（オプション）

        Returns:
            self
        """
        self._target_asset = target_asset

        # 特徴量とターゲットを生成
        features = self.create_features(prices, volumes, target_asset)
        target = self.create_target(prices, target_asset)

        # 欠損値を除去
        valid_mask = features.notna().all(axis=1) & target.notna()
        X = features[valid_mask]
        y = target[valid_mask]

        if len(X) < 50:
            raise ValueError(
                f"Insufficient data for training: {len(X)} samples"
            )

        # スケーリング
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # モデル選択と学習
        if self.config.model_type == "lightgbm" and not self._use_fallback:
            self._model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
            )
        else:
            self._model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_scaled, y)

        self._is_fitted = True

        logger.info(
            "ReturnPredictor fitted: asset=%s, model=%s, n_features=%d, "
            "n_samples=%d, horizon=%d",
            target_asset,
            self.config.model_type if not self._use_fallback else "random_forest",
            len(self._feature_names),
            len(X),
            self.config.prediction_horizon,
        )

        return self

    def predict(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        return_details: bool = False,
    ) -> Union[float, PredictionResult]:
        """リターンを予測

        Args:
            prices: 価格データ（最新データを含む）
            volumes: ボリュームデータ（オプション）
            return_details: 詳細結果を返すか

        Returns:
            予測リターン（float）または PredictionResult
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # 最新データで特徴量生成
        features = self.create_features(prices, volumes, self._target_asset)

        # 最新行を取得
        latest = features.iloc[-1:]

        if latest.isna().any().any():
            logger.warning("Some features contain NaN values")
            # NaNを0で埋める
            latest = latest.fillna(0)

        # スケーリングと予測
        X_scaled = self._scaler.transform(latest)
        prediction = float(self._model.predict(X_scaled)[0])

        if return_details:
            return PredictionResult(
                prediction=prediction,
                feature_values=latest.iloc[0].to_dict(),
                timestamp=prices.index[-1] if isinstance(prices.index, pd.DatetimeIndex) else None,
            )

        return prediction

    def predict_series(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """時系列で予測

        Args:
            prices: 価格データ
            volumes: ボリュームデータ

        Returns:
            予測値のSeries
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = self.create_features(prices, volumes, self._target_asset)

        # 欠損値を処理
        valid_mask = features.notna().all(axis=1)
        X = features[valid_mask].fillna(0)

        X_scaled = self._scaler.transform(X)
        predictions = self._model.predict(X_scaled)

        result = pd.Series(np.nan, index=prices.index)
        result[valid_mask] = predictions

        return result

    def get_feature_importance(self) -> pd.Series:
        """特徴量重要度を取得

        Returns:
            特徴量名をインデックスとした重要度のSeries（降順）
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        )
        return importance.sort_values(ascending=False)

    @property
    def is_fitted(self) -> bool:
        """モデルが学習済みかどうか"""
        return self._is_fitted

    @property
    def feature_names(self) -> List[str]:
        """特徴量名のリスト"""
        return self._feature_names.copy()


class MultiAssetPredictor:
    """複数アセット予測クラス

    複数のアセットに対してReturnPredictorを適用する。

    Usage:
        multi_predictor = MultiAssetPredictor()
        multi_predictor.fit(prices_df, assets=["AAPL", "MSFT", "GOOGL"])
        predictions = multi_predictor.predict(prices_df)
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        prediction_horizon: int = 20,
        **kwargs: Any,
    ) -> None:
        """初期化

        Args:
            model_type: モデルタイプ
            prediction_horizon: 予測期間
            **kwargs: ReturnPredictorに渡すパラメータ
        """
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.kwargs = kwargs
        self._predictors: Dict[str, ReturnPredictor] = {}
        self._is_fitted = False

    def fit(
        self,
        prices: pd.DataFrame,
        assets: Optional[List[str]] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> "MultiAssetPredictor":
        """全アセットのモデルを学習

        Args:
            prices: 価格データ
            assets: 対象アセット。Noneの場合は全カラム
            volumes: ボリュームデータ

        Returns:
            self
        """
        if assets is None:
            assets = list(prices.columns)

        for asset in assets:
            logger.info("Fitting predictor for %s", asset)
            predictor = ReturnPredictor(
                model_type=self.model_type,
                prediction_horizon=self.prediction_horizon,
                **self.kwargs,
            )
            try:
                predictor.fit(prices, asset, volumes)
                self._predictors[asset] = predictor
            except Exception as e:
                logger.warning("Failed to fit predictor for %s: %s", asset, e)

        self._is_fitted = True
        return self

    def predict(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """全アセットの予測を取得

        Args:
            prices: 価格データ
            volumes: ボリュームデータ

        Returns:
            {asset: prediction} の辞書
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions: Dict[str, float] = {}
        for asset, predictor in self._predictors.items():
            try:
                predictions[asset] = predictor.predict(prices, volumes)
            except Exception as e:
                logger.warning("Prediction failed for %s: %s", asset, e)
                predictions[asset] = 0.0

        return predictions

    def get_ranked_assets(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """予測リターンでランク付けしたアセットを取得

        Args:
            prices: 価格データ
            volumes: ボリュームデータ
            top_n: 上位N件のみ返す

        Returns:
            [(asset, prediction), ...] のリスト（降順）
        """
        predictions = self.predict(prices, volumes)
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        if top_n is not None:
            ranked = ranked[:top_n]

        return ranked


# ============================================================
# 便利関数
# ============================================================

def create_return_predictor(
    model_type: str = "lightgbm",
    prediction_horizon: int = 20,
    **kwargs: Any,
) -> ReturnPredictor:
    """ReturnPredictorを作成

    Args:
        model_type: モデルタイプ
        prediction_horizon: 予測期間
        **kwargs: その他のパラメータ

    Returns:
        ReturnPredictor インスタンス
    """
    return ReturnPredictor(
        model_type=model_type,
        prediction_horizon=prediction_horizon,
        **kwargs,
    )


def predict_returns(
    prices: pd.DataFrame,
    target_asset: str,
    prediction_horizon: int = 20,
    volumes: Optional[pd.DataFrame] = None,
) -> float:
    """リターンを予測（ワンライナー）

    Args:
        prices: 価格データ
        target_asset: 予測対象アセット
        prediction_horizon: 予測期間
        volumes: ボリュームデータ

    Returns:
        予測リターン
    """
    predictor = create_return_predictor(prediction_horizon=prediction_horizon)
    predictor.fit(prices, target_asset, volumes)
    return predictor.predict(prices, volumes)


def get_return_features(
    prices: pd.DataFrame,
    asset: str,
    lookback_features: Optional[List[int]] = None,
) -> pd.DataFrame:
    """特徴量のみを取得（分析用）

    Args:
        prices: 価格データ
        asset: 対象アセット
        lookback_features: ルックバック期間リスト

    Returns:
        特徴量DataFrame
    """
    predictor = ReturnPredictor(lookback_features=lookback_features)
    return predictor.create_features(prices, asset=asset)
