+++
title = "我们实现了单事实 OCR 遗漏检测的“标准答案”——它没有奏效，而失败的原因更有价值"
description = "版面模型覆盖率信号——用版面检测器找到的区域对照实际填充它们的 OCR 块打分——在与几何基线相同的两套真值数据集上测量：在签名/印章标注区上取得 {{ layout_gedi_ratio }}× 的遮挡分离度（我们测过的最锐利信号），但在单事实遗漏上点二列相关仅 {{ layout_fin_pointbiserial }}，低于它本要超越的几何基线 {{ cov_fact_pointbiserial }}。"
lang = "zh"
slug = "layout-coverage.zh"
date = "2026-07-12"
target_queries = [
  "版面分析 OCR 漏识别",
  "OCR 漏字检测 版面模型",
  "文档版面检测 覆盖率信号",
  "OCR 遗漏 布局区域",
]
[[faq]]
q = "版面检测模型能找出 OCR 静默丢失的文字吗？"
a = "只在区域尺度上可以。用版面检测器的区域对照填充它们的 OCR 块打分，是一个优秀的遮挡检测器——签名/印章标注区内的未覆盖区域率为 {{ layout_gedi_uncov_in }}，区外仅 {{ layout_gedi_uncov_out }}（{{ layout_gedi_ratio }}× 分离度）——但它无法可靠标记单个被丢掉的数字：对照专家事实级真值，点二列相关仅 {{ layout_fin_pointbiserial }}，低于便宜得多的几何墨迹覆盖率检查的 {{ cov_fact_pointbiserial }}。"
[[faq]]
q = "为什么区域级覆盖率检测不到单个丢失的数字？"
a = "因为被丢掉的数字通常位于一个 OCR 仍然部分填充了的区域内。版面检测器画出一个文本区域；OCR 读出了其中大部分内容、只静默丢掉一个值；该区域的填充率仍高于任何合理阈值，于是它被算作已覆盖。更细的几何粒度救不了这一点——失败的本质是：在任何几何粒度上，部分填充看起来都和完全填充一样。"
[[faq]]
q = "那到底什么能检测单事实 OCR 遗漏？"
a = "在我们的测量中，任何覆盖率形态的信号都不能。几何墨迹信号和版面模型信号都是区域尺度的遮挡检测器，不是事实级的遗漏检测器。诚实的剩余候选是值级校验：数值交叉验证（必须相加成立的合计、必须可解析的日期）、对低填充区域做第二遍 OCR、或用两个独立引擎互相比对——验证的是值本身，而不是几何。"
[[howto]]
step = "安装：git clone 仓库，uv sync --extra dev，再安装 OCR/版面栈：uv pip install paddleocr paddlepaddle"
[[howto]]
step = "先构建被评分的 OCR 缓存：uv run python -m contract_rag.eval.fincritical 与 uv run python -m contract_rag.eval.realscan（数据集需申请/外部下载，永不入库）"
[[howto]]
step = "运行版面覆盖率验证：uv run python -m contract_rag.eval.layout_coverage（版面推理按页落盘缓存，重跑很快）"
+++

# 我们实现了单事实 OCR 遗漏检测的“标准答案”——它没有奏效，而失败的原因更有价值

**TL;DR：** 在[OCR 遗漏一文](/ocr-omission.zh.html)中我们展示过：{{ fin_omission_rate }} 的专家标注事实从 OCR 输出中消失，而文档级质量分却是 {{ fin_quality_score }}；几何墨迹覆盖率能抓住区域尺度的丢失，却抓不住单个丢失的数字（点二列相关 {{ cov_fact_pointbiserial }}）。当时我们点名了那个显而易见的更细粒度方案：**版面模型覆盖率**——用版面检测器找到的区域，对照实际填充它们的 OCR 块打分。现在我们把它实现并在同样两套真值数据集上测完了。结果：它是**我们测过的最锐利的遮挡检测器**（签名/印章标注区 {{ layout_gedi_ratio }}× 分离度，几何信号是 {{ cov_zone_ratio }}×）——但它**没有通过它为之而生的单事实测试**：点二列相关 **{{ layout_fin_pointbiserial }}**，*低于*几何基线的 {{ cov_fact_pointbiserial }}。文中所有数字均为定点测量值（**{{ layout_measured_date }}**），提交在 `content/layout_results.toml`，可用文末命令复现。

## 假设

几何信号的弱点有一个干净的解释：一个被丢掉的财务数字只是几千个像素中的几个，几乎撼不动整页的未覆盖墨迹比率。自然的修法是提高粒度：不再问“这一*页*的墨迹是否都有着落”，而是问“版面检测器找到的每个*区域*是否真的被 OCR 块填充了”。区域尺寸的问题，应该能注意到区域尺寸的空洞。

于是我们实现了它：PaddleOCR 的 `LayoutDetection`（PP-DocLayout_plus-L）逐页提出版面区域；每个区域按其面积被 OCR 块包围盒覆盖的比例打分（`fill_ratio`）；低于填充阈值的区域计为未覆盖，文档的 `layout_omission_score` 就是未被填充区域的占比。推理按页落盘缓存；信号是纯增量的——质量报告上新增两个可选字段，质量分本身逐字节不变。

## 结果一：我们测过的最锐利的遮挡信号

在 {{ layout_gedi_pages }} 页带专家签名/印章区域标注的真实扫描件（Tobacco800）上：

| | 标注区内 | 区外 |
|---|---|---|
| 平均区域填充率 | **{{ layout_gedi_fill_in }}** | {{ layout_gedi_fill_out }} |
| 未覆盖区域率 | **{{ layout_gedi_uncov_in }}** | {{ layout_gedi_uncov_out }} |

未覆盖区域率的分离度达到 **{{ layout_gedi_ratio }}×**——约为几何墨迹信号 {{ cov_zone_ratio }}× 的两倍，且 {{ layout_gedi_pages_lower }} 页（{{ layout_gedi_pages_pct }}）的区内填充率更低。对合同档案迁移中真正要紧的问题——[这份文件到底有没有实体签署过](/signature-detection.zh.html)、印章是否正在被静默丢弃——版面区域正是路由人工复核的正确粒度，明显优于我们之前的任何信号。

如果标题只到这里，这会是一篇报捷文。但它不是，因为遮挡场景本来就不是那个悬而未决的问题。

## 结果二：事实级缺口没有合拢

悬而未决的问题是单事实遗漏——那 {{ fin_omission_rate }} 静默消失的专家标注事实。与几何信号完全相同的测量，在同样 {{ layout_fin_n_pages }} 页退化 SEC 文件上（其中 {{ layout_fin_pages_omitted }} 页至少含一个被遗漏的金标事实）：

| 信号 | 点二列相关（是否有遗漏） | 结论 |
|---|---|---|
| 几何墨迹覆盖率（基线） | {{ cov_fact_pointbiserial }} | 弱 |
| 版面模型覆盖率（“修复方案”） | **{{ layout_fin_pointbiserial }}** | **更弱** |

均值差距其实*拉大了*——含遗漏页的平均版面遗漏分 {{ layout_fin_mean_omitted }}，干净页仅 {{ layout_fin_mean_clean }}（约 {{ layout_fin_mean_ratio }}×，几何信号约 {{ cov_fact_ratio }}×）——但逐页相关性被噪声淹没（皮尔逊 {{ layout_fin_pearson }}）。均值差距更宽、相关性却更差，意味着这个信号太频繁地在错误的页面上触发，无法用于路由。

## 为什么更细的几何粒度救不了它

这次否证比数字本身更有用，因为它把机制磨得更锋利了。被丢掉的数字通常并不住在版面检测器标为空的区域里，而是住在一个 OCR **部分**填充了的区域里：检测器画出一个文本块，OCR 读出其中大部分，其中一个值静默消失。该区域的填充率仍高于任何合理阈值，于是被算作已覆盖。

这否掉的不只是这一个方案，而是整个方案家族。在*任何*几何粒度上——整页、区域、行——部分填充看起来都和完全填充一样，因为失败根本不是空间性的。丢失的信息是一个值，只有对值本身做推理的东西才能察觉它的缺席：数值交叉验证（必须相加成立的合计、必须闭合的百分比）、对低填充区域做定向二遍 OCR、或用两个独立引擎互相比对。覆盖率信号——无论几何还是版面模型——就是区域尺度的遮挡检测器，仅此而已。

## 对你的管线意味着什么

- **如果你本来就在跑版面模型，用版面覆盖率做遮挡路由**：{{ layout_gedi_ratio }}× 的分离度是我们测过最好的签名/印章丢失路由器。如果没有版面模型，几何墨迹检查零模型依赖也能给你 {{ cov_zone_ratio }}×。
- **不要把“版面感知的遗漏检测”当作事实级保证来买单。** 我们把这个显而易见的版本对照专家事实级真值测了，它表现*低于*纯几何基线。对任何宣称相反结论的供应商，索要他们对照事实级真值的点二列相关，而不是一场演示。
- **把单事实遗漏当作值验证问题，而不是覆盖率问题。** 我们自己的路线图接下来也走向那里。

## 诚实的局限

- **只测了一个版面模型、一组检测配置。** PP-DocLayout_plus-L 默认阈值；换检测器数字会变，但“部分填充”机制与检测器无关。
- **FinCriticalED 是 SEC 财务页面，不是商业合同**——原文的迁移性告诫原样适用。
- **填充阈值（0.5）未经调优。** 对着同一套真值调阈值就是过拟合评测；我们报告未调优的默认值。
- **定点测量数字（{{ layout_measured_date }}）**，以数据形式提交、构建时注入——因为数据集需申请/外部下载，永不入库。

## 自己复现

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
uv pip install paddleocr paddlepaddle
# 先构建 OCR 缓存（数据集需申请/外部下载——见 OCR 遗漏一文）：
uv run python -m contract_rag.eval.fincritical
uv run python -m contract_rag.eval.realscan
# 再跑版面覆盖率验证（版面推理按页落盘缓存）：
uv run python -m contract_rag.eval.layout_coverage
```

如果你的数字有实质性差异，请开 issue——负面结果只有在保持为真时才有价值。
