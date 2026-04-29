# demo_PhenoBrain

本仓库用于在本地运行和验证 `PhenoBrain` 离线诊断流程，并提供一个基于 `uv` 和 JupyterLab 的可复现实验环境。

## 仓库结构

- `timgroup_disease_diagnosis/`
  诊断核心代码、测试、数据读取器、集成模型实现和离线示例脚本。
- `scripts/start_jupyter_lab.sh`
  使用仓库内 `.jupyter` 配置启动 JupyterLab。
- `docs/plans/`
  设计文档、实现计划和完成记录。
- `ralph/`
  Ralph 任务拆分与执行状态。

## 环境要求

- Python `>=3.12`
- `uv`
- Java
  `BOQAModel` 需要本地 Java 运行 `boqa.jar`

## 快速开始

安装依赖：

```bash
uv sync
```

启动 JupyterLab：

```bash
./scripts/start_jupyter_lab.sh
```

## 离线集成模型

当前离线入口脚本：

- `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

> 注：`timgroup_disease_diagnosis/README.md` 仍保留上游研究项目说明；当前裁剪后的离线运行范围以本仓库根目录 `README.md` 为准。

命令行示例：

```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core \
.venv/bin/python \
timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py \
  --topk 5 \
  --hpo-list HP:0001913,HP:0008513,HP:0001123
```

命令行参数（与脚本 `--help` 一致）：

- `--topk`：返回数量，默认 `5`
- `--hpo-list`：逗号分隔的 HPO 列表；不传时使用内置样例：
  `HP:0001913,HP:0008513,HP:0001123,HP:0000365,HP:0002857,HP:0001744`

脚本会：

- 动态检查当前可用模型
- 构建外层 `Ensemble`
- 在启动时打印 `Available models: ...`
- 输出安静的结果表（列为 `Rank`、`Disease`、`Score`，`Score` 保留 6 位小数）

当前外层总集成已改为 **order-statistic / Stuart** 融合，显示的 `Score` 列是外层 `-log(Z)` 融合分数。

补充说明：

- 当可用模型数为 `1` 时，直接返回该模型，不构建外层集成
- 当可用模型数大于 `1` 时，构建 `OrderStatisticMultiModel(model_name='Ensemble')`
- 当无可用模型时，抛出 `RuntimeError: No diagnosis models are available for offline prediction.`

### 当前候选模型

- `ICTODQAcross-Ave-Random`
- `HPOProbMNB-Random`
- `CNB-Random`
- `NN-Mixup-Random-1`
- `MICAModel`
- `MICALinModel`
- `MICAJCModel`
- `MinICModel`
- `RBPModel`
- `GDDPFisherModel`
- `BOQAModel`

其中：

- 内层小集成仍使用原有 `OrderedMultiModel`
- 只有最外层 `Ensemble` 使用 order-statistic 融合

候选模型的可用性判定（代码逻辑）：

- `ICTODQAcross-Ave-Random`、`HPOProbMNB-Random`、`MICAModel`、`MICALinModel`、`MICAJCModel`、`MinICModel`、`RBPModel`、`GDDPFisherModel` 为基线候选
- `CNB-Random` 需要 `CNB.joblib`（兼容两种目录布局）
- `NN-Mixup-Random-1` 需要 TensorFlow checkpoint 文件（兼容两种目录布局）；若缺少 `core.predict.ml_model` 依赖会被自动跳过
- `BOQAModel` 需要同时满足：`java` 可执行、`boqa.jar` 存在、HPO 原始文件可用（优先 2019 路径，缺失时回退到 2022 路径）

## 数据与模型资产

`timgroup_disease_diagnosis/codes/core/model/` 下的模型文件用于本地离线推理，但已经加入忽略规则，不作为当前仓库版本控制的一部分。

这意味着：

- 仓库代码会依赖这些本地模型资产
- 如果缺少相关资产，部分候选模型会被自动跳过
- `BOQAModel` 依赖本地 Java 和 HPO 原始数据

仓库中仍有一部分超过 `10MB` 的非模型文件通过 Git LFS 管理。

## 作为 Library 迁移到其他项目

推荐把本仓库作为离线诊断 library 迁移时，采用“代码包 + 外部资产目录”的方式，不要只复制 Python 文件。

### 迁移 SOP

1. **固定可运行版本**
   - 在本仓库先跑通离线预测和 MME 金标准回归测试。
   - 记录 Python、`uv.lock`、TensorFlow、SciPy、Java 等运行环境。
2. **拆分代码与资产**
   - 代码迁移到目标项目的 `libs/phenobrain_diagnosis/` 或独立 package。
   - `data/`、`model/` 等大文件资产放在外部资产目录，由配置传入。
3. **封装稳定 API**
   - 目标项目不要直接依赖深层 `core.*` 内部路径。
   - 建议封装 `DiagnosisEngine.single(...)` 和 `DiagnosisEngine.ensemble(...)` 两类入口。
4. **参数化资产路径**
   - 通过初始化参数或环境变量传入 `data_dir`、`model_dir`。
   - 避免在目标项目中依赖当前仓库的绝对路径。
5. **集中处理兼容层**
   - `NN-Mixup-1` 是 TensorFlow 1 风格模型，在 TensorFlow 2 环境下需要 `tf.compat.v1` 兼容层。
   - 新版 SciPy 缺少 `scipy.stats.binom_test` 时需要兼容 shim。
6. **启动前做资产完整性检查**
   - 检查关键 JSON、NPZ、Numpy、Joblib、TensorFlow checkpoint 文件存在。
   - 检查文件不是 Git LFS pointer；如果文件头是 `version https://git-lfs.github.com/spec/v1`，说明资产未下载完整。
7. **标准化输入输出**
   - 输入统一为 HPO ID 列表，例如 `['HP:0008773', 'HP:0000413']`。
   - 输出建议包含 `RD:*`、对应 `SOURCE_CODES`（如 `OMIM:*`、`ORPHA:*`）和 score。
   - 对模型词表外 HPO 做过滤，并返回 `ignored_hpo_terms` 方便排查。
8. **迁移后跑回归测试**
   - 使用 `data/inputs/MME.benchmark_patients.questions.json` 和 `data/inputs/MME.benchmark_patients.answers.json` 验证 top-k 指标。
   - 迁移后的 rank/top-k 结果应与本仓库基准一致或只有可解释的微小差异。

### 推荐外部资产检查清单

- `timgroup_disease_diagnosis/codes/core/data/preprocess/knowledge/HPO/dis_to_hpo_prob_hpoa.json`
- `timgroup_disease_diagnosis/codes/core/data/preprocess/knowledge/disease-mix/rd_dict.json`
- `timgroup_disease_diagnosis/codes/core/model/INTEGRATE_CCRD_OMIM_ORPHA/ICTODQAcrossModel/ICTODQAcross-Ave/dis_vec_mat.npz`
- `timgroup_disease_diagnosis/codes/core/model/INTEGRATE_CCRD_OMIM_ORPHA/CNBModel/CNB.joblib`
- `timgroup_disease_diagnosis/codes/core/model/INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1/config.json`
- `timgroup_disease_diagnosis/codes/core/model/INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1/model.ckpt.index`
- `timgroup_disease_diagnosis/codes/core/model/INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1/model.ckpt.data-00000-of-00001`

### MME 回归基准

当前 MME 金标准评估集包含 `43` 个有答案病例。四模型外层集成使用：

- `ICTODQAcross-Ave`
- `HPOProbMNB`
- `CNB`
- `NN-Mixup-1`

融合方式为 `OrderStatisticMultiModel`。

| Model | top1 | top3 | top5 | top10 | top30 |
|---|---:|---:|---:|---:|---:|
| Ensemble(ICTODQAcross-Ave, HPOProbMNB, CNB, NN-Mixup-1) | 21/43 (0.4884) | 31/43 (0.7209) | 33/43 (0.7674) | 35/43 (0.8140) | 39/43 (0.9070) |

### GA4GH 全量基准

当前 GA4GH 全量评估集包含 `384` 个有答案病例。运行命令：

```bash
bash scripts/run_benchmark.sh
```

| Model | top1 | top3 | top5 | top10 | top30 |
|---|---:|---:|---:|---:|---:|
| Ensemble(ICTODQAcross-Ave, HPOProbMNB, CNB, NN-Mixup-1) | 1/384 (0.0026) | 5/384 (0.0130) | 7/384 (0.0182) | 12/384 (0.0312) | 20/384 (0.0521) |

### 通用 benchmark CLI

```bash
uv run python scripts/benchmark.py
uv run python scripts/benchmark.py --dataset MME --model MICAModel --ensemble none --metrics top1,top10
uv run python scripts/benchmark.py --dataset GA4GH --ensemble HPOP-ICT-CNB-NN --metrics top1,top3,top5,top10,top30
```

默认行为：运行所有可用单模型 + `HPOP-ICT-CNB-NN` 集成模型，数据集为 `MME,GA4GH`，指标为 `top1,top3,top5,top10,top30`。

## 验证

当前聚焦测试命令：

```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core \
.venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py \
  timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py \
  timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py \
  timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py \
  timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py \
  timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py \
  tests/test_pruned_runtime_layout.py \
  tests/test_jupyter_setup.py -q
```

## 当前限制

- 模型文件不随仓库自动获取，需本地自行准备
- `BOQAModel` 依赖 Java
- 如果远端不支持或未启用 Git LFS，含大文件的推送会失败
