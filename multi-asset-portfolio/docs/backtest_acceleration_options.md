# バックテスト高速化オプション調査報告書

**作成日**: 2026年1月30日

---

## 1. 現状分析

### 現在の処理時間
| 頻度 | リバランス回数 | 推定時間（現状） |
|------|---------------|-----------------|
| 月次 | 約180回 | 1-2分 |
| 週次 | 約780回 | 5-10分 |
| 日次 | 約3,900回 | 30-60分 |

### 現在実装済みの高速化技術

| 技術 | 効果 | 状態 |
|------|------|------|
| Numba JIT | 8-12倍 | ✅ 有効 |
| シグナル事前計算 | 5-10倍 | ✅ 有効 |
| インクリメンタル共分散 | 3-5倍 | ✅ 有効 |
| ProcessPool並列化 | 4-8倍 | ✅ 有効 |
| Polars DataFrame | 2-3倍 | ✅ 有効 |

### 未使用だが実装済みの高速化技術

| 技術 | 期待効果 | 状態 |
|------|---------|------|
| GPU計算（CuPy） | 10-50倍 | ⚠️ デフォルト無効 |
| Numba並列化 | 4-8倍 | ⚠️ デフォルト無効 |
| Ray分散処理 | 4-8倍 | ⚠️ デフォルト無効 |
| VectorBT完全ベクトル化 | 100-500倍 | ⚠️ テストレベル |

---

## 2. ローカル高速化オプション（即時実行可能）

### オプション A: GPU有効化

**概要**: CuPy/CUDAによるGPU計算

**設定変更**:
```python
config = FastBacktestConfig(
    use_gpu=True,  # 現在False
)
```

**期待効果**: 10-50倍高速化（共分散計算等）

**要件**:
- NVIDIA GPU（GTX 1060以上推奨）
- CUDA 11.x以上
- CuPyインストール

**コスト**: ハードウェア依存（既存GPU利用なら無料）

**参考**: [NVIDIA GPU-Accelerated Python](https://developer.nvidia.com/how-to-cuda-python)

---

### オプション B: Numba並列化の完全有効化

**概要**: CPUマルチコアの完全活用

**設定変更**:
```python
config = FastBacktestConfig(
    use_numba=True,
    numba_parallel=True,  # 現在False
)
```

**期待効果**: 追加4-8倍（コア数依存）

**要件**: Numba 0.50以上

**コスト**: 無料

---

### オプション C: Ray分散処理

**概要**: Python GIL制約を回避した真の並列処理

**設定変更**:
```python
from src.backtest.ray_engine import RayBacktestEngine
engine = RayBacktestEngine(n_workers="auto")
```

**期待効果**: CPUコア数に線形スケール（8コア→8倍）

**要件**: Ray 2.x

**コスト**: 無料

---

### オプション D: VectorBT完全ベクトル化（要開発）

**概要**: ループを完全排除したベクトル演算

**現状**: `vectorbt_engine.py` がテストレベルで存在

**期待効果**: 100-500倍（理論値）

**課題**: 複雑な戦略ロジックの実装が必要

**開発工数**: 中〜大

---

### ローカル高速化まとめ

**推奨設定（GPU有り）**:
```python
config = FastBacktestConfig(
    use_numba=True,
    numba_parallel=True,
    use_gpu=True,
    precompute_signals=True,
    use_incremental_cov=True,
)
# 期待効果: 40-150倍高速化
# 日次バックテスト: 30-60分 → 30秒-1分
```

**推奨設定（GPU無し）**:
```python
config = FastBacktestConfig(
    use_numba=True,
    numba_parallel=True,
    use_gpu=False,
    precompute_signals=True,
    use_incremental_cov=True,
)
# Ray併用で追加4-8倍
# 期待効果: 20-60倍高速化
```

---

## 3. クラウド高速化オプション

### オプション E: AWS GPU インスタンス

**概要**: クラウドGPUを使用したバックテスト

**推奨インスタンス**:
| インスタンス | GPU | vCPU | メモリ | 料金（東京） |
|-------------|-----|------|--------|-------------|
| g4dn.xlarge | T4 x1 | 4 | 16GB | $0.71/時 |
| g4dn.2xlarge | T4 x1 | 8 | 32GB | $1.06/時 |
| p3.2xlarge | V100 x1 | 8 | 61GB | $4.19/時 |

**期待効果**:
- g4dn.xlarge: 20-50倍高速化
- p3.2xlarge: 50-100倍高速化

**月額コスト目安**:
- 週1回1時間利用: $3-17/月
- 毎日1時間利用: $21-126/月

**実装方法**:
```bash
# EC2インスタンス起動後
pip install cupy-cuda11x
python scripts/run_backtest.py -f daily
```

---

### オプション F: AWS Lambda + Dask/Ray クラスター

**概要**: サーバーレス分散処理

**2025年の新機能**:
- **Lambda Durable Functions**: 長時間ワークロードの一時停止・再開
- **Lambda Managed Instances**: 予測可能なパフォーマンス

**アーキテクチャ**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Trigger    │────→│   Lambda    │────→│    S3       │
│ (EventBridge)│     │ (Dask/Ray)  │     │  (Results)  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                    ┌─────┴─────┐
                    │   EFS     │
                    │ (キャッシュ) │
                    └───────────┘
```

**期待効果**: 並列度に応じた線形スケール

**月額コスト目安**:
- Lambda: $0.20/100万リクエスト + 実行時間
- 週1回の15年バックテスト: 約$5-20/月

**参考**: [Dask on AWS](https://cloudprovider.dask.org/en/latest/aws.html)

---

### オプション G: Coiled（マネージドDask）

**概要**: Daskクラスターのマネージドサービス

**特徴**:
- ワンコマンドでクラスター起動
- 自動スケーリング
- スポットインスタンス対応

**料金**:
- フリープラン: 10,000 CPU時間/月
- Pro: $0.05/CPU時間

**実装例**:
```python
import coiled
cluster = coiled.Cluster(n_workers=10, spot_policy="spot_with_fallback")
# 10ワーカーで10倍高速化
```

**参考**: [Coiled Documentation](https://docs.coiled.io/)

---

### オプション H: 既存SaaS活用

**QuantConnect**:
- 料金: 無料〜$20/月
- 特徴: LEANエンジン、並列バックテスト
- 制約: 独自システムへの移植が必要

**Blueshift**:
- 料金: 無料（基本機能）
- 特徴: Python環境、教育向け
- 制約: インド・米国市場中心

**参考**: [QuantConnect](https://www.quantconnect.com/)

---

## 4. 比較表

| オプション | 高速化倍率 | 初期コスト | 月額コスト | 実装工数 |
|-----------|-----------|-----------|-----------|---------|
| A: GPU有効化 | 10-50倍 | GPU依存 | 無料 | 小 |
| B: Numba並列 | 4-8倍 | 無料 | 無料 | 極小 |
| C: Ray分散 | 4-8倍 | 無料 | 無料 | 小 |
| D: VectorBT | 100-500倍 | 無料 | 無料 | 大 |
| E: AWS GPU | 20-100倍 | 無料 | $20-130 | 小 |
| F: Lambda+Dask | 10-100倍 | 無料 | $5-50 | 中 |
| G: Coiled | 10-100倍 | 無料 | $0-100 | 小 |
| H: SaaS | - | 無料 | $0-20 | 大 |

---

## 5. 推奨アプローチ

### Phase 1: 即時実施（コスト: 無料、工数: 1日）

1. **Numba並列化を有効化**
   ```python
   numba_parallel=True
   ```

2. **既存GPU設定を有効化**（GPU有りの場合）
   ```python
   use_gpu=True
   ```

**期待効果**: 20-60倍高速化（日次バックテスト: 1-3分）

---

### Phase 2: 短期実施（コスト: 低、工数: 1週間）

1. **Ray分散処理の有効化**
2. **シグナル事前計算の完全実装**
3. **デフォルト設定の最適化**

**期待効果**: 追加2-4倍（累計40-150倍）

---

### Phase 3: 中期実施（コスト: 中、工数: 1ヶ月）

**選択肢A: クラウドGPU**
- AWS g4dn.xlarge を週次/月次で利用
- 月額$20-50程度

**選択肢B: Coiledマネージド**
- フリープランで開始
- 大規模時はProプランに移行

**期待効果**: オンデマンドで100倍以上

---

### Phase 4: 長期実施（コスト: 高、工数: 2-3ヶ月）

1. **VectorBT完全ベクトル化の本番化**
2. **Lambda Durable Functionsによるサーバーレス化**
3. **インクリメンタル更新の実装**

**期待効果**: 500倍以上（日次バックテスト: 数秒）

---

## 6. 技術的補足

### RAPIDS / cuDF

GPUでpandas互換の処理が可能:
```python
import cudf
df = cudf.read_parquet("prices.parquet")
# pandas と同じAPIでGPU計算
```

**参考**: [RAPIDS AI](https://rapids.ai/)

### NVIDIA cuTile Python（2025年12月リリース）

新しいGPUプログラミングモデル:
- タイル単位での処理（スレッド管理不要）
- 自動最適化
- 将来のアーキテクチャへのポータビリティ

**参考**: [cuTile Python](https://engineering.01cloud.com/2025/12/26/nvidia-cutile-python-simplifying-gpu-programming-for-the-next-generation/)

### gQuant（NVIDIA）

金融向けGPU加速ライブラリ:
- 5000銘柄のバックテストで20倍高速化
- RAPIDS上に構築

**参考**: [gQuant](https://medium.com/rapids-ai/gquant-gpu-accelerated-examples-for-quantitative-analyst-tasks-8b6de44c0ac2)

---

## 7. まとめ

**即座に実行可能な改善**:
- Numba並列化有効化で4-8倍
- GPU有効化で10-50倍
- 合計20-60倍（設定変更のみ）

**クラウド活用時**:
- AWS GPU: $20-130/月で20-100倍
- Coiled: 無料〜で10-100倍

**最終目標**:
- 日次15年バックテスト: 30-60分 → **数秒〜1分**
- 月次15年バックテスト: 1-2分 → **数秒**

---

## 関連ドキュメント

| ドキュメント | 概要 |
|-------------|------|
| [DEPLOYMENT.md](DEPLOYMENT.md) | 本番環境でのGPU/リソース設定 |
| [INSTALLATION.md](INSTALLATION.md) | GPU/Ray依存関係のインストール |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | パフォーマンス問題の解決 |
| [CACHE_SYSTEM.md](CACHE_SYSTEM.md) | キャッシュによる高速化 |
