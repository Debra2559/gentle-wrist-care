# 数据集调研

## 选择标准

项目需要同时覆盖两类目标：一是验证前臂与手部双节点原始 IMU 处理链路，二是评估腕屈伸和桡尺偏角度误差。优先条件为公开许可、腕部动作、前臂与手部传感器、原始惯性信号、同步参考角和可复现标注。

没有单个公开数据集同时完美满足全部条件，因此采用“主数据集 + 光学参考样例 + 后续候选”的组合策略。

## 工程接入状态

截至 2026-08-29，数据源不再通过训练脚本硬编码，而由 `config/datasets.json` 统一登记。每个条目包含 `dataset_id`、本地根目录、适配器、专家任务、来源标签映射、必需模态、许可、允许用途和证据限制。窗口数据保留 `dataset_id`，多来源合并时参与者、会话和序列 ID 会增加来源命名空间，避免跨数据集 ID 碰撞。

| 数据源 | 本地状态 | 当前代码能力 | 不可扩大解释 |
| --- | --- | --- | --- |
| Upper-body movements | `ready` | 五分类构建、按参与者留一、来源级指标 | 无独立腕角真值 |
| Comparison IMU vs Optotrak | `sample_only` | 单参与者来源工具箱 FE/RUD 误差报告 | 不是本项目原始 IMU 输出，也不是完整 16 人结果 |
| ULTRA-MoCap | `not_installed` | 角度专家契约占位 | 尚无本地数据或已核验字段映射 |
| OpenPack、LARa | `not_installed` | 工业场景专家契约占位 | 标签不能直接当作五类腕部动作 |
| MyoKi | `not_installed` | ADL 专家契约占位 | 无目标手背刚性 IMU |
| SheWrist 有线试采 | `unlabeled` | 域审计、解析兼容和故障测试 | 不可监督训练或确定融合权重 |

统一五类标签本体固定为 `background / extension / flexion / radial_deviation / ulnar_deviation`。只有标签和特征语义兼容的数据集才允许联合构建活动窗口；异构任务必须保持独立专家。当前只有一个兼容标注活动集，所以留一数据集评估会明确输出 `not_evaluable`。

专家融合接口已经预留，但 `validated_weights=null`。代码要求融合权重必须绑定一份带参与者、会话、CAL、动作和重戴身份的目标硬件验证集及指标名称；没有该证据时不允许按数据量、主观经验或简单平均生成权重。

运行审计：

```bash
.venv/bin/python scripts/audit_ml_datasets.py
.venv/bin/python scripts/evaluate_cross_dataset_activity.py
```

## 已纳入数据

### Upper-body movements

- 来源：Zenodo，DOI `10.5281/zenodo.4029127`。
- 许可：`CC BY 4.0`。
- 内容：11 名健康参与者，4 个 IMU，包含胸部、右上臂、右前臂和右手。
- 本项目使用：`set2` 的腕屈伸、桡尺偏和解剖中立位。
- 实际数据：100 Hz；加速度、角速度、磁场三轴原始流；共有 793 个传感器文本文件。
- 本地压缩包 MD5：`c5cd11bd72bccbebfe713ff95618bfc4`，与来源记录一致。
- 优点：与双节点腕部算法高度匹配，可运行原始传感器融合、相对姿态和功能轴标定。
- 限制：只有动作时间区间标签，没有量角器或光学角度真值，因此不能计算绝对角度 MAE。
- 数据异常：`subject06` 一条 Radial Deviation 标注为 `19.34–54.65 s`，明显超过其余单动作区间。代码用 `max_interval_s=15` 排除此异常段，并保留另一条正常重复。

验证设计避免标签循环论证：每类动作第一个有效重复用于轴标定，后续重复留出做方向检查。

### Comparison IMU vs Optotrak

- 来源：Zenodo，DOI `10.5281/zenodo.10935873`。
- 许可：数据 `CC BY 4.0`；公开工具箱 MIT。
- 完整数据：16 名健康参与者、IMU 与 Optotrak 同步上肢动作，约 4.4 GB。
- 本项目已纳入：公开工具箱仓库中的单参与者样例及已对齐角度 pickle，没有下载完整数据。
- 优点：提供腕屈伸和桡尺偏的光学参考，可验证误差统计流程。
- 限制：当前脚本读取的是来源工具箱已计算的 IMU 角度，而不是本项目 Madgwick 原始传感器输出，因此结果只能称为“来源工具箱基线”。

## 候选数据

### ULTRA-MoCap

- DOI：`10.6084/m9.figshare.28751156.v1`。
- 许可：`CC BY 4.0`。
- 规模：13 名参与者，约 5.18 GB。
- 内容：Vicon、IMU、sEMG、肌骨模型关节角；6 个 IMU 覆盖手、前臂、上臂和躯干等位置。
- 价值：适合耦合上肢运动、跨速度、传感器到解剖坐标映射和 OpenSim 角度对照。
- 限制：五类任务主要针对肩肘协调，并非孤立腕屈伸和桡尺偏主训练集。

### MyoKi

- DOI：`10.6084/m9.figshare.28696778`。
- 许可：`CC BY 4.0`。
- 内容：35 名参与者、74 项现实日常任务、每项 6 次；12 通道肌电、9 个六轴 IMU 和 CyberGlove 18 个手指/腕关节角，部分参与者含 FMG。
- 价值：适合日常活动下的腕角分布、动作迁移、跨人泛化和多模态扩展。
- 限制：IMU 分布于前臂和上臂，没有手背刚性 IMU，因此不能直接验证 `q_rel = inv(q_forearm) × q_hand` 的双节点腕角算法。

### WatchHand

- 数据 DOI：`10.7298/qf1v-j805`。
- 内容：40 名参与者，单只智能手表 IMU、声学和视频派生手部真值，包含多设备、左右手、跨会话重戴、姿势和噪声条件。
- 价值：适合参考重戴、跨设备、跨手和环境噪声的鲁棒性评测协议。
- 限制：不是前臂加手背双 IMU 数据，也不是 SheWrist 几何腕角验收集。

### Ninapro DB9

- DOI：`10.5281/zenodo.3354437`，当前 Zenodo 记录可见更新版本。
- 内容：77 名完整肢体参与者，40 种手部动作和抓握，CyberGlove II 提供 22 个校准关节角。
- 价值：适合腕和手指角度分布、动作分类、拇指模块扩展。
- 限制：主要是数据手套角度，并非当前前臂/手背双 IMU 原始融合验证。

### VIDIMU

- 概念 DOI：`10.5281/zenodo.7681316`；公开版本记录 `10.5281/zenodo.7681317`。
- 许可：`CC BY 4.0`。
- 内容：54 名视频参与者，其中 16 名同步使用 5 个 IMU，13 项日常活动，含 OpenSim 关节角。
- 价值：适合日常活动和视频/IMU多模态评估。
- 限制：公开原始 IMU 主要为设备内部融合四元数，不包含可重跑本项目姿态滤波所需的完整原始加速度、角速度和磁场；也缺少金标准光学系统。

### MoCap and IMU Dataset for Upper Limb Joint Angle Estimation

- Zenodo 记录：`10.5281/zenodo.15778950`，总量约 `456 MB`。
- 内容：同步 MoCap 与两个 IMU，但节点位于腕表位和胸部，目标主要是肩肘关节角。
- 结论：下载成本低但任务与节点不对口，不列入 SheWrist 腕角主验证序列。

## 结论

当前主数据集用于跑通双 IMU 原始处理链路，但没有独立角度真值。统一注册、来源追踪、专家契约、缺失拒绝和留一数据集评估入口已经实现；真正的多数据集训练尚未发生，因为本地只有一个标签兼容的活动集。下一阶段数据获取顺序固定为：

1. 完整 `Comparison IMU vs Optotrak`：第一优先，用本项目算法从原始 IMU 重算，并对 Optotrak 腕屈伸、桡尺偏计算独立 MAE。
2. `ULTRA-MoCap`：第二优先，先取处理后关节角和必要原始分包，用于耦合运动、跨速度及坐标映射验证。
3. `MyoKi`：先取 3–5 名参与者做适配和数据量评估，再决定是否扩展；用于日常任务分布和泛化，不用于双 IMU 几何真值。
4. `WatchHand`：只采用其跨会话重戴、跨设备和噪声评测协议。

这些公开数据均不能证明目标硬件、压力安全、De Quervain 疗效或疾病风险。

## 引用链接

- [Upper-body movements](https://zenodo.org/records/4029127)
- [Comparison IMU vs Optotrak](https://zenodo.org/records/10935873)
- [Biomechanics joint-angle toolbox](https://github.com/alexbonf/Biomechanics-joint-angle-analysis)
- [ULTRA-MoCap](https://doi.org/10.6084/m9.figshare.28751156.v1)
- [MyoKi](https://figshare.com/articles/dataset/28696778)
- [WatchHand](https://github.com/witlab-kaist/WatchHand)
- [Ninapro DB9](https://zenodo.org/records/3354437)
- [VIDIMU](https://zenodo.org/records/7681317)
- [MoCap and IMU upper-limb dataset](https://zenodo.org/records/15778950)