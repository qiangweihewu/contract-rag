+++
title = "“传统”OCR 引擎才是最爱幻觉的那个——而质量分完全看不见"
description = "我们把两个视觉语言 OCR 模型（dots.ocr、DeepSeek-OCR）拉来对照 PaddleOCR，测专家级事实遗漏和一个全新的幻觉指标。视觉 OCR 在几乎每个维度都赢——{{ vo_fin_dots_number_omitted }}/{{ vo_fin_dots_number_n }} 个数字遗漏、面对垃圾输入能安全失败——但预注册的采纳门槛依然判定 FAIL（{{ vo_fin_dots_omission }} 对比 {{ vo_rubric_bar }} 的门槛），同日完成的双引擎交叉核验也证实“跑两个 OCR 引擎”并不能简单解决问题（召回率仅 {{ vo_cc_recall }}）。"
lang = "zh"
slug = "vision-ocr.zh"
date = "2026-07-14"
target_queries = [
  "视觉语言模型 OCR 对比传统 OCR 基准",
  "dots.ocr 合同识别准确率",
  "OCR 幻觉检测",
  "DeepSeek-OCR 遗漏",
]
[[faq]]
q = "视觉语言 OCR 模型遗漏的事实比 PaddleOCR 少吗？"
a = "在我们的测量中是的：dots.ocr 在 FinCriticalED 上的整体遗漏率（{{ vo_fin_dots_omission }}）优于 PaddleOCR（{{ vo_fin_paddle_omission }}）；而对合同/财务事实抽取管线最要紧的数字类事实，dots.ocr 在 {{ vo_fin_dots_number_n }} 个中只遗漏了 {{ vo_fin_dots_number_omitted }} 个，PaddleOCR 的数字遗漏率则是 {{ vo_fin_paddle_number }}。相反，DeepSeek-OCR 在每一项遗漏指标上都比传统引擎更差（整体 {{ vo_fin_dsocr_omission }}，数字遗漏 {{ vo_fin_dsocr_number }}）。"
[[faq]]
q = "既然视觉 OCR 模型几乎每项都赢，为什么 PaddleOCR 还是默认引擎？"
a = "因为采纳决策是在数字出来之前就预先注册好的，就是为了防止我们事后给自己找理由。门槛要求 FinCriticalED 遗漏率不高于 PaddleOCR 遗漏率的一半——即 {{ vo_rubric_bar }}——而 dots.ocr 的 {{ vo_fin_dots_omission }} 没有达标。其余的表现（field-F1、退化时的安全失败行为、数字遗漏接近零）让 dots.ocr 成为我们测过的最值得推荐的可选 `VLM_ENDPOINT` 候选，但仍不是默认项。"
[[faq]]
q = "第二个 OCR 引擎能抓住第一个悄悄丢掉的东西吗？"
a = "在我们同日完成的测量中，不算可靠。把 PaddleOCR 的输出与 dots.ocr 的输出交叉比对，找出主引擎缺失的关键数字类 token，整体事实级召回率只有 {{ vo_cc_recall }}，未达到预注册的 {{ vo_cc_verdict }} 门槛——因为我们真值数据里大多数被遗漏的事实是纯文字的实体名，数字 token 设计天生就看不见它们。但在它本来就是为之设计的数字类事实上，表现好得多：抓住了 {{ vo_cc_number_caught }} 个被遗漏的数字，占全部数字类遗漏的 {{ vo_cc_digit_recall }}，误报率仅 {{ vo_cc_false_alarm }}。"
[[howto]]
step = "安装：git clone 仓库，uv sync --extra dev，再按 scripts/measure_vision_ocr.py 的运行手册装好 OCR/VLM 栈（在租用的 GPU 上用 vLLM 部署 dots.ocr / DeepSeek-OCR）"
[[howto]]
step = "先构建被评分的 OCR 缓存：uv run python -m contract_rag.eval.fincritical 与 uv run python -m contract_rag.eval.degrade（数据集需申请/外部下载，永不入库）"
[[howto]]
step = "运行视觉 OCR 测量：uv run python scripts/measure_vision_ocr.py --model dots（再 --model dsocr）"
[[howto]]
step = "运行双引擎交叉核验（离线，仅用已缓存的 IR）：uv run python -m contract_rag.eval.crosscheck"
+++

# “传统”OCR 引擎才是最爱幻觉的那个——而质量分完全看不见

**TL;DR：** 我们发过的每一篇负面结果文章都指向同一个根因：PaddleOCR 会静默*遗漏*内容，而下游的任何信号——置信度、几何墨迹覆盖率、版面区域覆盖率——都无法可靠捕捉到它。这次我们问了一个不同的问题：一个逐页整体阅读、而不是拼接检测+识别两步的现代视觉语言 OCR 模型，能否真正修复这个遗漏盲点？我们在同样的 FinCriticalED 事实级真值和同样的退化阶梯上，测了两个候选模型（dots.ocr、DeepSeek-OCR）对照 PaddleOCR，外加一个全新的幻觉指标，专门测视觉模型更容易踩的那个失败方向。结果：视觉 OCR 模型在我们测的几乎每一项上都赢——却依然没通过我们在跑测量之前就固定好的预注册采纳门槛。以下所有数字均为定点测量值（**{{ vo_measured_date }}**，{{ vo_gpu }}），提交在 `content/vision_ocr_results.toml`，可用文末命令复现。

## 我们为什么跑这个测量

这个系列此前每一篇文章的模式都一样：文档级质量分读数接近满分（FinCriticalED 上 **{{ fin_quality_score }}**），而 **{{ fin_omission_rate }}** 的专家标注事实从 OCR 输出中彻底消失。OCR 置信度无法标记它（一个被遗漏的事实根本不产生任何块，而不是产生一个低置信度的块）。几何墨迹覆盖率和版面区域覆盖率都能抓住区域尺度的遮挡（签名、印章），却抓不住单个被丢掉的数字。我们试过的每一个修复方案，治的都是*症状*——PaddleOCR 那种先检测、再拼接识别区域、途中丢掉一些内容的管线结构。

视觉语言 OCR 模型是把整页内容一次性读完的——版面、识别、阅读顺序一起处理——所以它没有那个会在拼接检测区域时丢内容的接缝。它的失败模式理应是相反的：不是静默遗漏，而是自信地*编造*。这两个方向都能对照我们已经缓存好的真值数据来测量，所以不管结果如何都值得发表：要么这个视觉模型真正修复了遗漏盲点，要么这是第五个测得很扎实的负面结果。

## 结果一：FinCriticalED 遗漏——视觉模型获胜，尤其是在数字上

同样的 {{ fin_n_pages }} 页金标数据、{{ fin_n_facts }} 条专家事实、同样的遗漏打分方式——现在跑了两个视觉 OCR 候选模型：

| 引擎 | 整体遗漏率 | 数字类事实遗漏率 |
|---|---|---|
| PaddleOCR（基线） | {{ vo_fin_paddle_omission }} | {{ vo_fin_paddle_number }} |
| **dots.ocr** | **{{ vo_fin_dots_omission }}** | **{{ vo_fin_dots_number_omitted }}/{{ vo_fin_dots_number_n }}** |
| DeepSeek-OCR | {{ vo_fin_dsocr_omission }} | {{ vo_fin_dsocr_number }} |

dots.ocr 在两列上都胜过传统引擎，而数字类事实的结果才是真正的标题级发现：在 {{ vo_fin_dots_number_n }} 个数字类事实中，它一个都没有丢掉——这正是合同或财务事实抽取管线最依赖的一类。但这个优势仅限于“数字”这一类，并不能推广到所有带数字的事实：dots.ocr 在日期/时间类事实上仍然遗漏了 **{{ vo_fin_dots_temporal }}**，这是一个真实存在的残留缺口，不在“数字类事实零遗漏”这个结论的覆盖范围内。DeepSeek-OCR 则朝反方向走——不仅整体上比 PaddleOCR 差，数字遗漏更是差得离谱，我们认为这与它的光学压缩式方法恰好对最要紧的那类事实不友好有关，再叠加它硬性的上下文长度上限——这本身也是一个真实的服务侧限制。而这整个系列的起点、那个盲点，在这里依旧引擎无关：三个引擎的文档级质量分读数都是 **{{ vo_fin_quality_all }}**，所以质量分完全无法告诉你哪个引擎正在丢事实。

## 结果二：退化阶梯——以及一个刻画*另一个*失败方向的新指标

FinCriticalED 只能测量遗漏方向（金标事实在输出里缺席）。退化阶梯——把干净的数字版 CUAD 页面渲染、退化、再重新 OCR——则给了我们相反方向的真值，因为原始的干净文本本身就是参照。我们新增了 `invented_token_ratio`：OCR 输出 token 中在原始页面文本任何地方都找不到的比例，经过规范化处理，使纯格式差异（大小写、千位分隔符、货币符号）永远不计数，而误读的数字或符号则会计数。

同一份此前退化测量用过的 {{ vo_deg_n_docs }} 文档、前 {{ vo_deg_n_pages }} 页切片上的 field-F1：

| 等级 | PaddleOCR F1 | dots.ocr F1 |
|---|---|---|
| light | {{ vo_deg_f1_light_paddle }} | {{ vo_deg_f1_light_dots }} |
| medium | {{ vo_deg_f1_medium_paddle }} | {{ vo_deg_f1_medium_dots }} |

两个等级 dots.ocr 都赢。而真正带来重新框定的，是幻觉 token 指标——它把*自信的垃圾*和*安全的失败*分开了：在 `fax` 等级，PaddleOCR 编造了 **{{ vo_inv_fax_paddle }}** 比例的输出 token，而它自己的质量分仍读作 **{{ vo_quality_fax_paddle }}**；同一等级下 dots.ocr 只编造了 **{{ vo_inv_fax_dots }}**。到 `shred` 等级，PaddleOCR 编造比例达到 **{{ vo_inv_shred_paddle }}**，dots.ocr 是 **{{ vo_inv_shred_dots }}**（它基本上是什么都不返回，而不是胡编），而 DeepSeek-OCR 编造了 **{{ vo_inv_shred_dsocr }}**——严重幻觉，而它自己的质量分却盲目地读作 **{{ vo_quality_shred_dsocr }}**。**PaddleOCR，这个“传统”引擎，是我们测过的最爱幻觉的那个，而质量公式对此完全视而不见**——这与遗漏那个故事是同一个盲点，只是方向相反。

**这个指标本身的告诫：** 在内容稀薄的文档上——即便是那 {{ vo_deg_n_pages }} 页原始干净数字文本本身也几乎是空的——幻觉 token 比例会对每个引擎同等地被拉高，因为一份很短的参照文本会让几乎任何输出 token 都被算作“不在参照里”。上面 `light` 等级的绝对值主要就是被这种参照偏差效应主导的；在内容密集页面上，逐文档的数值接近于零。请把*引擎之间的差值*当作有意义的信号，而把*绝对数字*当作对参照偏差敏感、不宜直接引用。

还有一个塑造了整个测量工具的运维发现：不加限制运行时，两个视觉 OCR 模型在严重退化的页面上都会陷入**重复循环**——生成 **{{ vo_loop_tokens }}** 个垃圾 token，每页耗时 **{{ vo_loop_minutes }}** 分钟，我们后来加了硬性生成上限才解决。任何生产环境的 VLM-OCR 部署都需要这个上限，不是可选项。

## 预注册的门槛：没有通过

在上面任何数字出现之前，我们就固定了一条采纳门槛：视觉模型只有在其 FinCriticalED 遗漏率不高于 PaddleOCR 遗漏率的**一半**时——即 {{ vo_rubric_bar }}——才会成为默认的扫描路由引擎。dots.ocr 实测遗漏率为 {{ vo_fin_dots_omission }}，高于这个门槛。**结论：未通过。** PaddleOCR 依然是默认扫描路由引擎；VLM 路由保持可选。延迟数据也支持这一点——GPU 上 {{ vo_latency_vlm }} 对比 PaddleOCR 在 CPU 上的 {{ vo_latency_paddle }}，对一个连门槛都没过的方案来说是实打实的代价，不是舍入误差。但诚实的结论并不是“别用它”：在 field-F1、退化时的安全失败行为、数字类事实近乎零遗漏这几方面，dots.ocr 是我们测过的最强候选，值得作为可选的 `VLM_ENDPOINT`。DeepSeek-OCR 在这个垂直领域的任何一个维度上都不推荐。

## 结果三：第二个引擎能抓住第一个漏掉的东西吗？

考虑到 dots.ocr 在数字类事实上的优势，一个自然的下一个问题是：把它当作*校验者*而不是替代品来用。让 PaddleOCR 作为主引擎，把它的输出与 dots.ocr 的输出做差异比对，找出主引擎缺失的数字类“关键 token”（数字、金额、日期、百分比）——一旦对不上，就把该页标记为需要人工复核。同样，我们在跑之前就预先注册了门槛：{{ vo_cc_bar }}。

在同样缓存的 FinCriticalED IR 上离线测量：整体标记召回率 **{{ vo_cc_recall }}**，误报率 **{{ vo_cc_false_alarm }}**。**结论：{{ vo_cc_verdict }}**（门槛 {{ vo_cc_bar }}）——离召回率那一侧的目标还差得远。诚实地说，原因是设计本身的局限，而不是 bug：{{ vo_cc_entity_caught }} 个实体名遗漏被抓住——数字 token 交叉核验从设计上就不可能看见纯文字的实体名，而实体名遗漏正是数据集里 {{ vo_cc_n_omissions }} 个总遗漏中最大的一类。但在这个校验机制本来就是为之设计的数字类事实上，它确实有效：抓住了 **{{ vo_cc_number_caught }}** 个被遗漏的数字类事实，占全部数字类遗漏的 **{{ vo_cc_digit_recall }}**，误报率仍是那同一个 {{ vo_cc_false_alarm }}。**双引擎交叉核验是一个可用的数字类事实安全网，而不是一个通用的遗漏检测器**——这和覆盖率那组工作里“区域尺度对比事实级尺度、纯几何信号看不见值本身”的教训是同一个道理，只是这次换成了第二个 OCR 引擎而不是几何比率。

## 诚实的局限

- **FinCriticalED 是 SEC 财务文件，不是商业合同。** 遗漏率结果（结果一）和交叉核验的召回率/误报率数字（结果三）都是在 FinCriticalED 上测出的；它们作为*OCR 属性*可以迁移，但不是合同字段准确率的数字。只有退化阶梯的结果（结果二）是在合同页面（CUAD）上跑出来的。
- **DeepSeek-OCR-2 → v1 替换。** 设计方案原本瞄准 DeepSeek-OCR 2；实际运行时发现它的 `DeepseekOCR2ForCausalLM` 架构没有任何与实验机 CUDA 12.8 驱动兼容的 vLLM 构建，于是我们替换成了 DeepSeek-OCR v1。上面关于 v1 的数字不能作为 OCR-2 那个具体架构的证据。
- **退化是模拟的。** 退化阶梯是一个受控的、带种子的压力测试（缩放、倾斜、JPEG 重压缩、二值化、噪声）——不是野外采集来的真实脏数据。它和 `dirtify` 一样，是一个校准工具，而不是真实分布。
- **幻觉比例的绝对值带参照偏差。** 内容稀薄的文档会对每个引擎同等地拉高该比例（见上文）——应该在同一等级下比较引擎之间的差异，而不是跨等级比较原始数字。
- **重复循环是真实存在的，不加上限的生成不适合生产环境。** {{ vo_loop_tokens }} 个 token、每页 {{ vo_loop_minutes }} 分钟，是在没有生成上限的情况下发生的；任何 VLM-OCR 部署从第一天起就需要这个上限。
- **交叉核验在它被预注册对照的那个指标上是正式的 FAIL**，即便数字类事实那个拆分结果是真实的正面结果——我们按这个顺序如实同时报告两者。
- **单一 GPU 实验机、每个模型只跑了一次。** 视觉 OCR 的这些数字没有重复跑的方差估计（不像我们一些 LLM 抽取测量那样带 bootstrap 置信区间）。
- **定点测量数字（{{ vo_measured_date }}）**，以数据形式提交、构建时注入——因为 GPU 实验机是租用后即销毁的，数据集也需申请/外部下载，永不入库。

## 自己复现

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
# 先构建 OCR 缓存（数据集需申请/外部下载——见 OCR 遗漏一文）：
uv run python -m contract_rag.eval.fincritical
uv run python -m contract_rag.eval.degrade
# 视觉 OCR 测量需要一个已部署的 VLM 端点（见 scripts/measure_vision_ocr.py 的运行手册）：
uv run python scripts/measure_vision_ocr.py --model dots
uv run python scripts/measure_vision_ocr.py --model dsocr
# 双引擎交叉核验在两份 IR 缓存都存在后完全离线：
uv run python -m contract_rag.eval.crosscheck
```

如果你的数字有实质性差异，请开 issue——负面结果只有在保持为真时才有价值。
