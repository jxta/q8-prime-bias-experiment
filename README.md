# Q8 Galois Extension Prime Bias Experiment

青木美穂・小山新也論文「Chebyshev's Bias against Splitting and Principal Primes in Global Fields」(arXiv:2203.12266) のExample 2.1におけるQ8（第四四元数群）をガロア群とする体についての素数の偏りに関する数値実験環境です。

## 🎯 プロジェクト概要

このプロジェクトは、Quaternion拡大におけるChebyshevバイアスの数値実験を実行するための完全なPython環境を提供します。Omar S.論文の13ケース全てについて、10^9までの大規模計算に対応した高速化・並列化実装を含んでいます。

## ✨ 主要機能

- **13ケース完全対応**: Omar S.論文で定義された全13のQuaternion拡大について計算
- **高速化実装**: SymPyとNumPyを組み合わせた効率的なフロベニウス元計算
- **並列化処理**: マルチプロセシングによる大規模計算の高速化
- **実用的なスケール**: 10^9までの素数に対応した実装
- **5つのグラフ表現**: 理論的偏りと実験結果の包括的可視化
- **Jupyter Notebook対応**: SageMath, Julia, Pythonカーネルに対応

## 📦 インストール

```bash
git clone https://github.com/jxta/q8-prime-bias-experiment.git
cd q8-prime-bias-experiment
pip install -r requirements.txt
```

## 🚀 クイックスタート

### 1. 小規模テスト（10^4素数）

```python
from src.experiment_runner import QuaternionBiasExperiment

# 実験インスタンスを作成
experiment = QuaternionBiasExperiment()

# Case 1-3を10^4素数まで実行
experiment.run_cases([1, 2, 3], max_prime=10**4)

# 結果の可視化
experiment.generate_all_graphs()
```

### 2. 大規模実験（10^6-10^9素数）

```python
# 並列処理で大規模実験を実行
experiment.run_parallel_experiment(
    case_ids=list(range(1, 14)),  # 全13ケース
    max_prime=10**6,              # 10^6素数まで
    num_processes=8               # 8プロセス並列
)
```

### 3. Jupyter Notebookでのデモ

```bash
jupyter notebook notebooks/demo.ipynb
```

## 📂 プロジェクト構造

```
q8-prime-bias-experiment/
├── README.md
├── requirements.txt
├── LICENSE
├── config/
│   └── experiment_config.yaml    # 実験設定
├── src/
│   ├── __init__.py
│   ├── omar_polynomials.py       # 13ケースの多項式定義
│   ├── fast_frobenius_calculator.py  # 高速化フロベニウス計算
│   ├── bias_analyzer.py          # 偏り解析とグラフ生成
│   ├── experiment_runner.py      # 実験実行管理
│   └── utils.py                  # ユーティリティ関数
├── notebooks/
│   ├── demo.ipynb               # デモ用ノートブック
│   ├── advanced_analysis.ipynb  # 高度な解析用
│   └── performance_test.ipynb   # パフォーマンステスト
├── scripts/
│   ├── run_all_cases.py         # 全ケース実行スクリプト
│   ├── batch_analysis.py        # バッチ解析スクリプト
│   └── performance_benchmark.py # ベンチマークスクリプト
├── data/
│   └── results/                 # 計算結果保存ディレクトリ
├── graphs/
│   └── output/                  # グラフ出力ディレクトリ
└── tests/
    ├── test_calculations.py     # 計算テスト
    ├── test_polynomials.py      # 多項式テスト
    └── test_performance.py      # パフォーマンステスト
```

## 🔬 理論的背景

### Chebyshevバイアス

青木・小山の定理により、深層リーマン予想（DRH）の下で、フロベニウス元σに対する素数の偏りは：

```
π_{1/2}(x) - (8/|C_σ|)π_{1/2}(x;σ) ≈ (M(σ) + m(σ)) log(log(x))
```

ここで：
- `π_{1/2}(x)` : 重み付き素数計数関数
- `|C_σ|` : σの共役類のサイズ
- `M(σ) + m(σ)` : 偏り係数（m_ρ₀に依存）

### Q8群の構造

Quaternion群Q8 = {±1, ±i, ±j, ±k}は8つの元を持ち、以下の共役類に分けられます：

- C₁ = {1} (サイズ 1)
- C₋₁ = {-1} (サイズ 1)  
- Cᵢ = {i, -i} (サイズ 2)
- Cⱼ = {j, -j} (サイズ 2)
- Cₖ = {k, -k} (サイズ 2)

### 偏り係数

各ケースの`m_ρ₀`値に応じて、以下の偏り係数が期待されます：

**m_ρ₀ = 0の場合：**
- g₀(1): +1/2, g₁(-1): +5/2
- g₂(i), g₃(j), g₄(k): -1/2

**m_ρ₀ = 1の場合：**
- g₀(1): +5/2, g₁(-1): +1/2  
- g₂(i), g₃(j), g₄(k): -1/2

## 📊 実験結果の解釈

### 5つのグラフ表現

1. **S1**: π_{1/2}(x) - 8π_{1/2}(x;1) vs log(log(x))
2. **S2**: π_{1/2}(x) - 8π_{1/2}(x;-1) vs log(log(x))
3. **S3**: π_{1/2}(x) - 4π_{1/2}(x;i) vs log(log(x))
4. **S4**: π_{1/2}(x) - 4π_{1/2}(x;j) vs log(log(x))
5. **S5**: π_{1/2}(x) - 4π_{1/2}(x;k) vs log(log(x))

### 期待される傾向

- 赤い線: 理論値 (M(σ) + m(σ)) log(log(x))
- 黒い点: 実際の計算結果
- 一致度が高いほど、理論の妥当性が確認される

## ⚡ パフォーマンス

### 計算性能（参考値）

| 素数範囲 | シングルプロセス | 8プロセス並列 | 16プロセス並列 |
|----------|-----------------|---------------|----------------|
| 10^4     | ~0.5秒          | ~0.2秒        | ~0.2秒         |
| 10^5     | ~5秒            | ~1秒          | ~0.8秒         |
| 10^6     | ~50秒           | ~8秒          | ~5秒           |
| 10^7     | ~8分            | ~1.5分        | ~1分           |
| 10^8     | ~1.5時間        | ~15分         | ~10分          |
| 10^9     | ~15時間         | ~2.5時間      | ~1.5時間       |

### 高速化のポイント

1. **効率的なフロベニウス計算**: クロネッカー記号の最適化
2. **並列処理**: 素数範囲を分割してマルチプロセシング
3. **メモリ管理**: 中間結果の効率的な保存
4. **進捗表示**: tqdmによるリアルタイム進捗確認

## 🧪 使用例

### 基本的な使用例

```python
import sys
sys.path.append('./src')

from experiment_runner import QuaternionBiasExperiment
from omar_polynomials import get_case, print_all_cases_summary

# 全ケースの概要を表示
print_all_cases_summary()

# 特定のケースの詳細を確認
case1 = get_case(1)
print(f"Case 1: {case1.polynomial}")
print(f"m_ρ₀: {case1.m_rho0}")
print(f"分岐素数: {case1.ramified_primes}")

# 実験を実行
experiment = QuaternionBiasExperiment()
results = experiment.run_case(1, max_prime=10**4)

# 結果の解析
experiment.analyze_bias(1)
experiment.plot_case_graphs(1)
```

### 並列処理での大規模実験

```python
# 設定
config = {
    'max_prime': 10**6,
    'num_processes': 8,
    'batch_size': 50000,
    'save_intermediate': True
}

# 全13ケースを並列実行
results = experiment.run_parallel_experiment(
    case_ids=list(range(1, 14)),
    **config
)

# 結果の比較分析
experiment.compare_all_cases()
experiment.generate_summary_report()
```

### カスタム解析

```python
from bias_analyzer import BiasAnalyzer

# カスタム解析器を作成
analyzer = BiasAnalyzer()

# 特定の偏り係数について詳細分析
for case_id in [1, 2, 3]:
    bias_data = analyzer.compute_bias_evolution(case_id, max_prime=10**5)
    analyzer.plot_bias_convergence(case_id, bias_data)
    analyzer.statistical_analysis(case_id, bias_data)
```

## 🔧 設定オプション

### experiment_config.yaml

実験の詳細設定は `config/experiment_config.yaml` で管理されます：

```yaml
computation:
  small_scale: 10000        # 10^4 for quick tests
  medium_scale: 100000      # 10^5 for development
  large_scale: 1000000      # 10^6 for research
  huge_scale: 1000000000    # 10^9 for production
  
  parallel:
    enabled: true
    max_processes: null     # auto-detect CPU cores
    batch_size: 50000       # primes per batch

visualization:
  graphs:
    figsize: [18, 12]
    dpi: 150
    style: "seaborn-v0_8"
```

## 🧪 テスト

```bash
# 全テストを実行
python -m pytest tests/

# 特定のテストを実行
python -m pytest tests/test_calculations.py -v

# パフォーマンステスト
python scripts/performance_benchmark.py
```

## 📝 引用

このコードを研究で使用する場合は、以下を引用してください：

```bibtex
@article{aoki2022chebyshev,
  title={Chebyshev's Bias against Splitting and Principal Primes in Global Fields},
  author={Aoki, Miho and Koyama, Shin-ya},
  journal={arXiv preprint arXiv:2203.12266},
  year={2022}
}
```

## 🤝 貢献

プルリクエストやissueの報告を歓迎します。特に以下の改善点について：

- フロベニウス元計算の更なる高速化
- 新しい可視化手法の追加
- 統計的有意性検定の実装
- 他のガロア群への拡張

## 📄 ライセンス

MIT License

## 👥 謝辞

このプロジェクトは青木美穂さんとの共同研究のために開発されました。理論的基盤となる Chebyshev バイアスの研究に深く感謝いたします。

---

**重要**: 大規模計算（10^8以上）を実行する前に、十分な計算資源があることを確認してください。実験の進捗はリアルタイムで保存されるため、中断しても途中から再開可能です。
