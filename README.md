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
