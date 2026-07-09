+++
title = "这份合同到底有没有实体签署过？在扫描档案中检测未签署文件"
description = "面向扫描文档的签名存在性检测器，在 {{ sig_n_pages }} 页真实 Tobacco800 文件上对照专家区域标注评分：精确率 {{ sig_precision }}、F1 {{ sig_f1 }}——而真正有用的是：在“默认都签过”假设一份也找不出的地方，它标出了 {{ sig_unsigned_total }} 份未签署文件中的 {{ sig_unsigned_flagged }} 份。"
lang = "zh"
slug = "signature-detection.zh"
date = "2026-07-09"
target_queries = [
  "扫描件签名检测",
  "如何判断合同是否已签署",
  "OCR 签名识别",
  "档案未签署合同检测",
]
[[faq]]
q = "如何检测一份扫描合同是否被实体签署过？"
a = "不是靠读出签名——OCR 读不出来——而是靠 OCR 输出中的三个块级信号：结尾敬辞（{{ sig_salutation_signed }} 的已签署文件带有，未签署文件只有 {{ sig_salutation_unsigned }}）、/s/ 或 By: 之类的显式签名线索，以及签名块——页面下部一行打字体人名、正上方压着一个低置信度乱码 token，这正是墨水签名在 OCR 引擎眼中的样子。三者合并后，在 {{ sig_n_pages }} 页真实扫描件上：精确率 {{ sig_precision }}、F1 {{ sig_f1 }}。"
[[faq]]
q = "OCR 能识别手写签名吗？"
a = "不能直接识别——OCR 引擎没有签名这个概念。但它在墨水签名上的失败模式本身就是可用的信号：引擎会在读得干干净净的打字体姓名（“{{ sig_typed_name }}”，置信度 {{ sig_typed_name_conf }}）正上方输出一个乱码低置信度 token（如“{{ sig_squiggle_text }}”，置信度 {{ sig_squiggle_conf }}）。这种“涂鸦压姓名”的几何模式能找回没有结尾敬辞的已签署备忘录和表单，在我们的评测集上误报成本为零。"
[[faq]]
q = "扫描文档签名检测的准确率有多高？"
a = "在 {{ sig_n_pages }} 页真实 Tobacco800 文件上（{{ sig_n_signed }} 签署 / {{ sig_n_unsigned }} 未签署，以专家区域标注为真值）：精确率 {{ sig_precision }}、召回率 {{ sig_recall }}、F1 {{ sig_f1 }}。重点在未签署一侧：标出 {{ sig_unsigned_total }} 份未签署文件中的 {{ sig_unsigned_flagged }} 份、仅 {{ sig_false_positives }} 个误报，而“默认都签过”基线一份也找不出。注意事项：启发式针对打字机信件惯例调优，召回率上限约 {{ sig_recall }}，因为约 1/4 的已签署文件是没有敬辞的表单或备忘录。"
[[howto]]
step = "获取数据：Tobacco800 页面 TIFF 及 GEDI XML 区域标注（签名/标志区域；来自 Illinois Complex Document Image Processing 集合——数据集文件从不提交到本仓库）"
[[howto]]
step = "安装：git clone 本仓库，uv sync --extra dev，扫描路径另需 paddleocr"
[[howto]]
step = "运行评测：SIGNATURE_DIR=path/to/tobacco800/tiffs SIGNATURE_GT_DIR=path/to/gedi/xml uv run python -m contract_rag.eval.signature（输出精确率/召回率/F1 及“默认都签过”基线对比；OCR 解析有 IR 缓存）"
+++

# 这份合同到底有没有实体签署过？在扫描档案中检测未签署文件

**核心结论：** 我们构建了一个面向扫描文档的签名存在性检测器，在 **{{ sig_n_pages }}** 页真实 Tobacco800 文件（{{ sig_n_signed }} 签署 / {{ sig_n_unsigned }} 未签署）上对照专家区域标注评分：**精确率 {{ sig_precision }}、召回率 {{ sig_recall }}、F1 {{ sig_f1 }}**。平凡基线——假设档案里每份合同都签过——F1 {{ sig_baseline_f1 }}，但找出的未签署文件数量是**零**；检测器标出了 **{{ sig_unsigned_total }} 份中的 {{ sig_unsigned_flagged }} 份**，只有 {{ sig_false_positives }} 个误报。最强的信号出乎意料：OCR *读不出*墨水签名，而这个失败本身正是我们检测的对象。所有数字均为定点实测（**{{ sig_measured_date }}**），提交在 `content/signature_results.toml` 中，文末给出复现命令。

## 签名在 OCR 引擎眼中长什么样

这是评测集里的一个真实签名，paddleocr 看到的样子——上世纪 70 年代一封打字机信件底部相邻的两个块：

| 块文本 | OCR 置信度 |
|---|---|
| `{{ sig_squiggle_text }}` | {{ sig_squiggle_conf }} |
| `{{ sig_typed_name }}` | {{ sig_typed_name_conf }} |

引擎把打字体姓名读得一字不差，却在上方的墨水涂鸦上栽了跟头——输出了一个乱码的低置信度 token。它完全不知道那里有个签名。但这个*模式*——页面下部、一行读得干干净净的人名、正上方压着一个低置信度乱码块——正是签名在 OCR 输出中留下的可靠痕迹。我们检测的不是签名，是签名的残骸。

## 谁需要这个

问题来自合同生命周期管理（CLM）：企业迁移遗留档案——成千上万份扫描的历史合同——需要知道**其中哪些从未真正签署过**。档案里躺着一份未签署的合同是法律和审计问题，而默认假设（"进了档案就是已签署"）对相当大的比例是静默错误的：我们这个真实档案评测集里，**{{ sig_n_pages }} 页中有 {{ sig_n_unsigned }} 页**根本没有签名。任何文档级质量分数都不会告诉你这件事——我们在[上一篇 OCR 遗漏文章](/ocr-omission.zh.html)里单独测过这个盲区：缺失的签名和任何遗漏一样，不产生任何可供质量信号打分的块。

## 三个信号，以及一个被刻意否决的

`detect_signature(ir)` 组合三个块级信号，每个都是从真实 Tobacco800 OCR 输出中设计出来的，以概率或（probabilistic OR）合并为 P(signed) 置信度：

1. **结尾敬辞**——匹配"Sincerely / Regards / Very truly yours / …"的块。遥遥领先的最强信号：**{{ sig_salutation_signed }}** 的已签署文件带有，未签署文件只有 **{{ sig_salutation_unsigned }}**，且精确率近乎完美。一封落了款的信就是签了的信。
2. **显式签名线索**——`/s/`、`By:`、"duly authorized"、"authorized signature"。
3. **签名块**——上文那个"涂鸦压姓名"的几何模式：页面下部一行打字体人名，正上方压着一个低置信度 OCR token。它能找回*没有*敬辞的已签署备忘录和表单，在本评测集上误报成本为零。

有一个诱人的信号被刻意**否决了**：把逐块 OCR 置信度当作通用的遮挡检测器。在真实扫描件上，与签名/印章区域重叠的块置信度确实更低（{{ realscan_conf_occluded }} vs {{ realscan_conf_elsewhere }}）——但未签署的传真和电传和已签署的信件一样噪声大，所以在块粒度上这个信号几乎不区分两者，依赖它反而伤精确率。置信度只有*锚定到几何*（信号 3）时才有用：姓名行正上方的低置信度有含义；泛泛的低置信度只说明这是张传真。

## 对照专家真值的结果

真值是 Tobacco800 的 GEDI 区域标注：一页算已签署**当且仅当**其标注带有签名区域。Tobacco800 对签名的标注是全面的，所以零区域的页面是*真正的*负例，不是"没标注"。在 {{ sig_n_pages }} 页上：

| | 精确率 | 召回率 | F1 | 准确率 |
|---|---|---|---|---|
| "默认都签过"基线 | {{ sig_baseline_precision }} | 1.000 | {{ sig_baseline_f1 }} | — |
| 签名检测器 | **{{ sig_precision }}** | {{ sig_recall }} | **{{ sig_f1 }}**（{{ sig_f1_delta }}） | {{ sig_accuracy }} |

F1 增量看起来不大，因为在一个大多已签署的语料上基线白拿满分召回。真正要看的是 F1 表不出来的那一列：基线从构造上就找不出**任何**未签署文件；检测器标出了 **{{ sig_unsigned_total }} 份中的 {{ sig_unsigned_flagged }} 份**，误报 {{ sig_false_positives }} 个。对档案迁移的那个问题——"把所有可能从未生效的文件都给我列出来"——这就是全部价值所在，而"默认都签过"一分价值也提供不了。

每个预测都带 `evidence_block_ids`，复核者能看到*哪些块*触发了判断——和流水线其余部分同样的来源归因纪律：无法追溯到证据的论断，是无法审计的论断。

## 诚实的局限

- **启发式针对 Tobacco800 的打字机信件惯例调优**——敬辞列表、姓名行正则、涂鸦几何。换语料（现代 DocuSign PDF、非英文档案）需要重新调优；这是语料校准的启发式，不是通用模型。
- **召回率上限 {{ sig_recall }}**：约 1/4 的已签署文件是没有敬辞的表单或备忘录，其签名也逃过了姓名块几何。训练一个分类器，或在更细粒度上使用区域级信号，可以把它提上去。
- **块粒度**：证据是被引用的 OCR 块，不是签名的像素区域。
- **这些是定点实测数字（{{ sig_measured_date }}）**，基于 {{ sig_n_pages }} 页的集合；数据集来自外部档案，从不提交到仓库。

## 自己复现

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
# Tobacco800 TIFF + GEDI XML 区域标注：来自 Illinois CDIP 集合（从不提交到本仓库）
SIGNATURE_DIR=path/to/tobacco800/tiffs SIGNATURE_GT_DIR=path/to/gedi/xml \
  uv run python -m contract_rag.eval.signature
```

OCR 解析有 IR 缓存，重跑很快。检测器本身是对解析后文档的纯逻辑——用手工构建的 IR 做单元测试，不需要 OCR 也不需要网络。如果你的数字有实质性差异，欢迎开 issue——公开测量的意义正在于此。
