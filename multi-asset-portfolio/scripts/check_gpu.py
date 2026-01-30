#!/usr/bin/env python3
"""
GPU環境確認スクリプト

GPU計算環境の有無を確認し、CuPyのインストール状況を報告する。

Usage:
    python scripts/check_gpu.py

GPU計算を有効にするには:
    1. CUDA対応GPUが必要
    2. CuPyをインストール:
       - CUDA 12.x: pip install cupy-cuda12x
       - CUDA 11.x: pip install cupy-cuda11x
    3. FastBacktestConfig.from_resource_config() でGPU自動検出
       または use_gpu=True を明示的に設定
"""

from __future__ import annotations

import sys


def check_gpu_environment() -> dict:
    """GPU環境を確認"""
    result = {
        "cupy_installed": False,
        "cupy_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "total_memory_gb": 0.0,
        "recommendation": "",
    }

    # CuPyのインストール確認
    try:
        import cupy
        result["cupy_installed"] = True
        result["cupy_version"] = cupy.__version__
    except ImportError:
        result["recommendation"] = (
            "CuPyがインストールされていません。\n"
            "GPU計算を使用するには以下をインストールしてください:\n"
            "  - CUDA 12.x: pip install cupy-cuda12x\n"
            "  - CUDA 11.x: pip install cupy-cuda11x\n"
            "  - CUDA 10.x: pip install cupy-cuda10x\n"
            "詳細: https://docs.cupy.dev/en/stable/install.html"
        )
        return result

    # CUDA確認
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count > 0:
            result["cuda_available"] = True
            result["device_count"] = device_count

            # CUDAバージョン
            cuda_version = cupy.cuda.runtime.runtimeGetVersion()
            result["cuda_version"] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

            # デバイス情報
            total_memory = 0
            for i in range(device_count):
                props = cupy.cuda.runtime.getDeviceProperties(i)
                mem_info = cupy.cuda.Device(i).mem_info
                device_info = {
                    "id": i,
                    "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                    "memory_total_gb": mem_info[1] / (1024**3),
                    "memory_free_gb": mem_info[0] / (1024**3),
                    "compute_capability": f"{props['major']}.{props['minor']}",
                }
                result["devices"].append(device_info)
                total_memory += mem_info[1]

            result["total_memory_gb"] = total_memory / (1024**3)
            result["recommendation"] = (
                "GPU環境が利用可能です。\n"
                "FastBacktestConfig.from_resource_config() で自動検出されます。\n"
                "手動で有効化する場合: FastBacktestConfig(use_gpu=True)"
            )
        else:
            result["recommendation"] = (
                "CuPyはインストールされていますが、GPUが検出されませんでした。\n"
                "CUDA対応GPUがシステムに接続されているか確認してください。"
            )
    except Exception as e:
        result["recommendation"] = f"GPU検出中にエラーが発生しました: {e}"

    return result


def print_gpu_report(result: dict) -> None:
    """GPU環境レポートを出力"""
    print("=" * 60)
    print("GPU環境確認レポート")
    print("=" * 60)

    # CuPy状態
    if result["cupy_installed"]:
        print(f"CuPy: インストール済み (v{result['cupy_version']})")
    else:
        print("CuPy: 未インストール")

    # CUDA状態
    if result["cuda_available"]:
        print(f"CUDA: 利用可能 (v{result['cuda_version']})")
        print(f"GPU数: {result['device_count']}")
        print(f"総GPUメモリ: {result['total_memory_gb']:.2f} GB")
        print()
        print("検出されたGPU:")
        for device in result["devices"]:
            print(f"  [{device['id']}] {device['name']}")
            print(f"      メモリ: {device['memory_total_gb']:.2f} GB "
                  f"(空き: {device['memory_free_gb']:.2f} GB)")
            print(f"      Compute Capability: {device['compute_capability']}")
    else:
        print("CUDA: 利用不可")

    print()
    print("-" * 60)
    print("推奨事項:")
    print(result["recommendation"])
    print("=" * 60)


def check_resource_config_gpu() -> None:
    """ResourceConfigのGPU検出状態を確認"""
    print()
    print("ResourceConfig GPU検出状態:")
    print("-" * 40)

    try:
        from src.config.resource_config import get_resource_config

        rc = get_resource_config()
        print(f"  use_gpu: {rc.use_gpu}")
        print(f"  gpu_memory_fraction: {rc.gpu_memory_fraction}")

        if rc.system_resources and rc.system_resources.gpu:
            gpu = rc.system_resources.gpu
            print(f"  検出GPU数: {gpu.device_count}")
            print(f"  総GPUメモリ: {gpu.total_memory_gb:.2f} GB")
    except ImportError:
        print("  (ResourceConfigのインポートに失敗)")
    except Exception as e:
        print(f"  エラー: {e}")


def main() -> int:
    """メイン関数"""
    result = check_gpu_environment()
    print_gpu_report(result)
    check_resource_config_gpu()

    # 終了コード: GPU利用可能なら0、そうでなければ1
    return 0 if result["cuda_available"] else 1


if __name__ == "__main__":
    sys.exit(main())
