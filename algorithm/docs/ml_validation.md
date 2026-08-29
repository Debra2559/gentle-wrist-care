# CNN-HMM 影子模型验证

验证日期：2026-08-29

## 结论

本项目已经实现并实际运行轻量 `1D-CNN + HMM`、置信度校准、质量与缺失数据拒识、故障增强、结构化惯性 Token，以及确定性安全链和机器学习影子链的并行分析。数据加载现由统一注册表驱动，支持来源身份、标签映射、专家契约与跨数据集评估入口。

当前模型的正确定位是：

> 公开健康参与者腕部四类动作的可复现基线，用于验证训练与推理管线。它不是劳损容量、疼痛、De Quervain 风险、压力安全或治疗需求模型。

模型必须保持 `shadow` 模式，不能触发或取消报警，不能控制机械装置。

## 数据与任务

数据来自 Upper-body movements `set2`：11 名健康参与者，前臂与手部 IMU，100 Hz 原始采样。算法重采样至 50 Hz，以既有双 IMU 管线生成腕屈伸、桡尺偏、角速度和质量信号。

五类标签为：

```text
background
extension
flexion
radial_deviation
ulnar_deviation
```

每名参与者每类动作有两个标注重复。第一个重复用于该参与者功能轴标定，并从分类窗口中排除；第二个重复用于分类评估。最终得到 2090 个窗口：

| 类别 | 窗口数 |
| --- | ---: |
| background | 1734 |
| extension | 93 |
| flexion | 97 |
| radial_deviation | 81 |
| ulnar_deviation | 85 |

窗口长度 `1.5 s`、步长 `0.5 s`，以中心样本分配标签，避免通过删除动作边界窗口向 HMM 泄漏边界信息。每个窗口同时保存 `dataset_id` 和 `missing_fraction`；多来源合并时参与者、会话和序列 ID 会增加来源命名空间。

## 模型

输入通道：

```text
theta_fe_deg
theta_rud_deg
dtheta_fe_deg_s
dtheta_rud_deg_s
angular_velocity_deg_s
quality
```

网络由单层时域卷积、ReLU、全局平均池化和 Softmax 构成，共 12 个卷积滤波器、卷积核长度 7。选择小模型是因为样本只有 11 人，扩大模型无法弥补数据和标签缺口。

HMM 只学习以下通用规律：

- 背景状态通常持续。
- 某一动作状态通常持续若干窗口。
- 动作可回到背景，背景可进入任一动作。
- 所有前景类别共享起始概率和转移先验。

因此 HMM 不记忆公开实验中的具体动作顺序。

训练增强包括幅值缩放、固定偏置、随机噪声、时间平移、短时缺失和单通道遮挡。Softmax 温度由验证参与者校准。窗口平均质量低于 `0.5`、缺失比例超过 `0.1` 或置信度低于 `0.55` 时输出 `unknown`。

## 评估设计

每一折使用：

```text
9 名参与者训练
1 名不同参与者验证、早停和温度校准
1 名参与者完全留出测试
```

11 名参与者轮流作为测试集。窗口从不随机跨人切分。由于测试参与者仍需先执行功能轴标定，本结果属于“校准辅助的跨人分类”，不是完全免校准泛化。

另提供 `leave-one-dataset-out` 评估入口：整份来源只允许出现在训练侧或测试侧，不能拆散混用。当前本地只有 `Upper-body movements` 一个标签与输入语义兼容的活动集，所以跨数据集评估状态为 `not_evaluable`；现有 11 折数字仍是单数据集跨参与者结果，不得写成跨数据集性能。

## 结果

| 指标 | 均值 | 标准差 | 最低 | 最高 |
| --- | ---: | ---: | ---: | ---: |
| CNN 准确率 | 0.597 | 0.156 | 0.220 | 0.783 |
| CNN macro-F1 | 0.534 | 0.176 | 0.164 | 0.749 |
| HMM 准确率 | 0.703 | 0.197 | 0.244 | 0.936 |
| HMM macro-F1 | 0.559 | 0.186 | 0.188 | 0.831 |
| 拒识后覆盖率 | 0.506 | 0.266 | 0.096 | 0.870 |
| 接受样本内准确率 | 0.664 | 0.285 | 0.193 | 1.000 |
| 拒识计入漏检的 macro-F1 | 0.477 | 0.189 | 0.062 | 0.834 |
| 事件 F1 | 0.587 | 0.223 | 0.167 | 0.889 |

HMM 对平均结果有帮助，但不能解决个体差异。最弱参与者折的 HMM macro-F1 仅 0.188，说明当前模型绝不能用于真实控制或产品宣称。

## 容错

| 测试扰动 | macro-F1 | 覆盖率 | 事件 F1 |
| --- | ---: | ---: | ---: |
| 无附加扰动 | 0.477 | 0.506 | 0.587 |
| 5% 随机缺失 | 0.474 | 0.485 | 0.564 |
| 10% 随机缺失 | 0.408 | 0.493 | 0.500 |
| 20% 随机缺失 | 0.365 | 0.545 | 0.450 |
| 20% 特征标准差随机噪声 | 0.480 | 0.497 | 0.572 |

该模型对所测试的独立随机噪声较稳，但对连续数据缺失更敏感。现实中的固定带滑移、轴错位、时间不同步和系统性传感偏差可能比独立噪声更严重，不能由本表覆盖。当前推理会把缺失比例超过 `10%` 的窗口直接拒识；这只是工程门控，不代表缺失数据已被可靠恢复。

## 多数据集与专家状态

`config/datasets.json` 当前登记 7 个来源。可执行状态为：

- `Upper-body movements`：11 人、2,090 窗口，五分类活动训练适配器已实现。
- `Comparison IMU vs Optotrak`：只接入单参与者来源工具箱样例，用于独立角度误差报告，不进入五分类投票。
- `ULTRA-MoCap`：预留上肢角度/耦合运动专家。
- `OpenPack + LARa`：预留工业工作场景专家。
- `MyoKi`：预留 ADL 上下文专家。
- SheWrist 有线试采：无标签，只用于域审计、兼容和故障测试。

未安装或未完成标签映射的专家只返回 `unavailable`，不会生成伪概率。概率融合要求 `ValidatedFusionWeights`，必须绑定目标硬件验证集 ID 和验证指标；当前 `validated_weights=null`，因此融合保持禁用。即使未来启用，输出仍为 `shadow`，控制权限为 `none`。

## 安全隔离

统一分析输出显式包含：

```text
angle_alert_authority = deterministic_exposure_engine
pressure_stop_authority = deterministic_pressure_channel
mechanical_action = manual_only
ml_control_authority = none
llm_control_authority = none
```

压力红区或用户不适绕过模型直接触发释放与停止。ML 低质量或低置信时只输出 `unknown`，不累计新的动作 Token，也不改变确定性暴露剂量。

## 运行

数据源与专家就绪度：

```bash
.venv/bin/python scripts/audit_ml_datasets.py
.venv/bin/python scripts/evaluate_cross_dataset_activity.py
```

完整训练：

```bash
PYTHONPATH=src .venv/bin/python scripts/train_activity_model.py \
  --dataset-id upper_body_movements \
  --output-dir outputs/ml
```

并行运行确定性和影子分支：

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_with_shadow.py data/processed/public_subject01_set2_joint_state.csv --model outputs/ml/activity_cnn_hmm_shadow.npz --session-id public-subject01 --evidence-type replay --output outputs/ml/combined_public_subject01.json
```

主要产物：

- `outputs/ml/summary.json`：精简指标与证据边界。
- `outputs/ml/loso_report.json`：逐折训练历史、混淆矩阵和故障注入结果。
- `outputs/ml/oof_predictions.csv`：全部测试折逐窗预测。
- `outputs/ml/oof_tokens.json`：由 OOF 接受结果生成的事件 Token。
- `outputs/ml/activity_cnn_hmm_shadow.npz`：全量公开数据训练模型。
- `outputs/ml/combined_public_subject01.json`：双分支统一输出示例。

## 下一步

要把活动模型升级出影子模式，至少还需要：第二个标签和输入语义兼容的数据集、目标硬件同步数据、多人多会话重戴、独立视频标签、长时背景任务、目标活动与 coping 事件定义，以及预注册的报警误报和漏报标准。完整 Optotrak、ULTRA-MoCap、OpenPack、LARa 或 MyoKi 到位后，必须先核验许可证、字段、传感器位置和标签本体，再启用对应适配器；不能只把文件放入目录就视为可训练。