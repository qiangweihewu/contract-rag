+++
title = "我们把爆火的新 OCR 引擎摆上了基准测试——然后留下了旧的"
description = "FrankenOCR（纯 Rust、纯 CPU 的 3B VLM OCR 封装）对 PaddleOCR，在我们自己的评测框架上：抽取打平（字段 F1 {{ fr_f1_light }} vs {{ fr_paddle_f1_light }}，n={{ fr_degrade_n }}，属噪声），慢 {{ fr_slowdown }}（{{ fr_sec_per_page }}/页 vs {{ fr_paddle_sec_per_page }}），以及一个比乱码更糟的失败模式：在不可读页面上自信地幻觉出结构——我们的质量分数把它读成 {{ fr_quality_shred }}。"
lang = "zh"
slug = "franken-ocr.zh"
target_queries = [
  "FrankenOCR PaddleOCR 对比",
  "FrankenOCR 评测",
  "VLM OCR 幻觉",
  "如何评估 OCR 引擎",
]
[[faq]]
q = "在文档处理流水线里 FrankenOCR 比 PaddleOCR 更好吗？"
a = "在我们的基准上不是。在退化的合同页面上两者抽取打平（字段 F1 {{ fr_f1_light }} vs {{ fr_paddle_f1_light }}，n={{ fr_degrade_n }}——噪声级差异），但 FrankenOCR 在 CPU 上慢 {{ fr_slowdown }}（{{ fr_sec_per_page }}/页 vs {{ fr_paddle_sec_per_page }}），且在不可读页面上不输出可见乱码，而是自信地幻觉出结构。我们把它作为环境变量开关的可选实验保留，默认引擎仍是 PaddleOCR。"
[[faq]]
q = "基于 VLM 的 OCR 引擎会产生幻觉吗？"
a = "会，而且这种失败模式比经典 OCR 乱码更糟：在一张刻意做到不可读的页面上，VLM 输出了空的表格骨架，以及一个解码耗时 {{ fr_hallucination_decode_time }} 的 {{ fr_hallucination_size }} 重复列表循环——流畅、格式良好、完全是编造的。我们的文档质量分数把这份输出读成 {{ fr_quality_shred }}，而 PaddleOCR 在同一页上的可见乱码只得 {{ fr_paddle_quality_shred }}。乱码会触发质量信号，流畅的幻觉却能骗过它们。"
[[faq]]
q = "该如何为自己的流水线评估一个新 OCR 引擎？"
a = "把它封装到与现有引擎相同的文档表示后面，让下游一切保持一致；然后在你自己的语料上做基准——包括最差的页面——测量端任务准确率（抽取 F1，而不只是文本相似度）、你的硬件上的每页速度，以及在不可读输入上的失败模式。演示截图展示的是最好情况；你日常运维的是失败模式。"
[[howto]]
step = "安装被测引擎：FrankenOCR（{{ fr_version }}，MIT，github.com/Dicklesworthstone/franken_ocr）——单个 {{ fr_binary_size }} 二进制，首次运行下载 {{ fr_weights_size }} 模型权重"
[[howto]]
step = "适配器已提交在本仓库：src/contract_rag/parse/franken_parser.py，通过 FRANKEN_BIN=path/to/focr 启用（不设置时解析路由字节级不变）"
[[howto]]
step = "运行基准：FRANKEN_BIN=path/to/focr uv run python scripts/benchmark_franken.py（realscan + degrade 两组；需要 Tobacco800 扫描件和 CUAD 黄金集——数据集从不提交；OCR 输出有 IR 缓存）"
+++

# 我们把爆火的新 OCR 引擎摆上了基准测试——然后留下了旧的

**核心结论：** FrankenOCR——纯 Rust、纯 CPU 的 3B 参数 OCR 视觉语言模型封装，近期市场热度的主角——被放上了我们自己的评测框架，对阵现任扫描文档引擎 PaddleOCR。结果：**抽取打平**（退化合同上字段 F1 **{{ fr_f1_light }}** vs **{{ fr_paddle_f1_light }}**，n={{ fr_degrade_n }}——噪声），**慢 {{ fr_slowdown }}**（同一 CPU 上 {{ fr_sec_per_page }}/页 vs {{ fr_paddle_sec_per_page }}），以及一个我们认为比经典 OCR 乱码*更糟*的失败模式：**在不可读页面上自信地幻觉出结构**——我们的质量分数把它读成满分 {{ fr_quality_shred }}。适配器作为可选实验留在仓库里；默认引擎不换。所有数字均为定点实测（**{{ fr_measured_date }}**），提交在 `content/franken_results.toml`，评测框架随仓库附带。

## 我们为什么要测它

FrankenOCR 有真实的吸引力，假装没有本身就是一种不诚实。它是单个 **{{ fr_binary_size }}** 二进制、零 Python 依赖（{{ fr_weights_size }} 权重首次运行时下载）——对封闭环境是实打实的运维卖点。它输出版面分类标签（`header` / `title` / `text` / `page_number` / `image`），我们的覆盖率实验将来用得上。而且它的热度已经大到"我们没看过"约等于失职。

于是我们做了对每个引擎都做的事：写一个适配器，收敛到所有解析器共用的同一 Document IR——每份文档一次子进程调用、markdown 输出重建为块、`FRANKEN_BIN` 环境变量按需启用、**不设置时路由行为字节级不变**——然后放上和现任引擎完全相同的评测框架。同样的文档、同样的抽取器、同样的指标。

## 数字

两组实验，运行于 {{ fr_measured_date }}：

| 实验组 | FrankenOCR | PaddleOCR |
|---|---|---|
| 速度（{{ fr_realscan_n }} 页真实 Tobacco800 扫描件，CPU） | **{{ fr_sec_per_page }}/页** | {{ fr_paddle_sec_per_page }}/页 |
| 抽取，degrade-light（{{ fr_degrade_n }} 份 CUAD 合同） | 字段 F1 {{ fr_f1_light }} | 字段 F1 {{ fr_paddle_f1_light }} |
| 抽取，degrade-shred（不可读） | {{ fr_f1_shred }} | {{ fr_f1_shred }} |

n={{ fr_degrade_n }} 时的抽取差异是噪声：按**打平**理解。速度差异不是噪声：同一硬件上 **{{ fr_slowdown }}**——放到档案迁移的规模，这是一夜跑完和跑一个季度的区别。

原始输出里有一个看似 FrankenOCR 获胜的数字，值得我们抢在别人引用之前自己拆穿：来源归因准确率 **{{ fr_srcacc_light }} vs {{ fr_paddle_srcacc_light }}**。这个差距是**块粒度的伪影**，不是准确率——FrankenOCR 每份文档只输出约 {{ fr_blocks_per_doc }} 个页面大小的 markdown 块，PaddleOCR 是 {{ fr_paddle_blocks_per_doc }} 个行级块；引用"就在这一页里"当然比引用正确的那一行容易命中。因为证据变粗而变好看的指标，不是变好了。

## 决定性的失败模式

在 `shred`——我们刻意做到不可读的退化档位，任何引擎都*应该*失败的地方——PaddleOCR 失败得很诚实：输出可见乱码，质量分数应声跌到 **{{ fr_paddle_quality_shred }}**，文档被标记复核。

FrankenOCR 的失败方式不同。它输出了**流畅、格式良好、完全编造的结构**：空的 `<table>` 骨架，还有一次是 **{{ fr_hallucination_size }}** 的"2. 3. 4. …"重复列表循环，解码耗时 **{{ fr_hallucination_decode_time }}**。我们的质量分数把这份输出读成 **{{ fr_quality_shred }}**——满分。

这正是[我们此前测过的质量信号盲区](/ocr-omission.zh.html)，只是更锋利的形态。质量信号衡量的是*引擎输出了的东西*：乱码会触发它们；沉默会绕开它们；而流畅的幻觉会主动**击败**它们——它恰恰制造出信号所奖励的那种格式良好的证据。对一条以*可溯源、可验证的法律文档事实*为产品的流水线来说，会在不可读输入上编造结构的引擎，无论吞吐量和打包形态如何，都没有资格坐默认位。乱码会被抓住；幻觉会被信任。

## 结论，以及背后的规则

FrankenOCR 作为**可选实验**留在仓库里——适配器和评测框架都已提交，它的版面标签将来也许能在覆盖率实验里挣回饭钱。PaddleOCR 仍是默认扫描文档引擎。

可迁移的不是结论，而是流程——每个候选引擎大约花一天：

1. **适配到同一内部表示**，让对比只隔离引擎本身（下游一切一致）。
2. **在你自己的语料上做基准**——包括你手里最差的页面，而不是演示里最好的。
3. **测端任务**（抽取 F1）、**你的硬件上的速度**，以及**不可读输入上的失败模式**——最后一项才是两个引擎真正分道扬镳的地方。

## 诚实的局限

- **样本小**：准确率组 {{ fr_degrade_n }} 份合同，速度组 {{ fr_realscan_n }} 页扫描件。对这个效应量级（{{ fr_slowdown }} 的速度差、一票否决的失败模式）的去留决策够用；不是排行榜。
- **纯 CPU 计时**，这是 FrankenOCR 自己的定位。GPU 部署会改变速度账——但不会改变幻觉发现。
- **单一版本**（{{ fr_version }}，{{ fr_measured_date }}）。VLM 封装迭代很快；评测框架被提交进仓库，正是为了重跑只需一条命令。
- **四个退化档位只跑了两个**（light、shred）——两个极端夹住了中间，但 medium/fax 的数字尚不存在。

## 自己复现

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
# FrankenOCR：{{ fr_binary_size }} 二进制，MIT——github.com/Dicklesworthstone/franken_ocr（首次运行下载 {{ fr_weights_size }} 权重）
FRANKEN_BIN=path/to/focr uv run python scripts/benchmark_franken.py
```

需要 Tobacco800 扫描件和 CUAD 黄金集（从不提交——见仓库 README）；OCR 输出有 IR 缓存，重跑很快。如果新版 FrankenOCR 改变了这些数字，跑一遍告诉我们——评测框架存在的意义，就是让这场争论可以用数据进行。
