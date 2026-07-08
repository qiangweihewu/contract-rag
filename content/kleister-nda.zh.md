+++
title = "Kleister-NDA 实测：公开基准上的诚实抽取数字"
description = "在公开的 Kleister-NDA 基准（真实 SEC EDGAR 保密协议）上的定点实测结果：规则抽取字段 F1 {{ kleister_f1_initial }} → {{ kleister_f1_improved }}，来源归因准确率 {{ kleister_source_acc }}；以及服务端 Schema 约束解码为何消除了 30% 的结构化输出失败。"
lang = "zh"
slug = "kleister-nda.zh"
target_queries = [
  "NDA 抽取基准测试",
  "Kleister NDA 测评结果",
  "合同结构化抽取准确率",
]
[[faq]]
q = "对真实 NDA 合同做结构化抽取的准确率有多高？"
a = "在公开的 Kleister-NDA 基准（确定性构建的 40 份真实 SEC EDGAR 保密协议）上，确定性的规则抽取器达到字段 F1 {{ kleister_f1_improved }}，来源归因准确率 {{ kleister_source_acc }}——测量时间为 2026-07-06，任何人拿一份 Kleister 数据集检出即可用一条命令复现。"
[[faq]]
q = "为什么合成数据的抽取演示会高估准确率？"
a = "因为规则和提示词就是在演示文档本身上调出来的。同一个抽取器在合成 NDA 集上得分 {{ synthetic_nda_f1 }}，在真实 Kleister-NDA 文档上（针对性修复前）只有 {{ kleister_f1_initial }}。合成数字证明的是流水线正确性；公开基准衡量的才是真实文档抽取能力。"
[[faq]]
q = "Schema 约束解码能解决 LLM 结构化输出失败吗？"
a = "在传输层面可以：在完全相同的环境（Ollama、qwen2.5:32b-instruct）下，服务端 response_format=json_schema 将 Schema 校验失败从 {{ tools_schema_failures }}（30%）降到 {{ constrained_schema_failures }}，字段 F1 从 {{ mlx_f1 }} 提升到 {{ constrained_f1 }}。但它无法修复错误引用片段——值正确、引用块却是错的。"
[[howto]]
step = "克隆数据集：git clone https://github.com/applicaai/kleister-nda（该仓库未声明明确许可证，因此本仓库从不提交数据集文件）"
[[howto]]
step = "构建确定性 40 文档集：KLEISTER_DIR=path/to/kleister-nda uv run python -m contract_rag.verticals.nda.kleister"
[[howto]]
step = "运行评测：KLEISTER_DIR=path/to/kleister-nda uv run python -m contract_rag.verticals.nda.kleister --eval（输出字段 F1 与来源归因准确率）"
+++

# Kleister-NDA 实测：公开基准上的诚实抽取数字

**核心结论：** 在公开的 **Kleister-NDA** 基准（arXiv 2105.05796——真实 SEC EDGAR 保密协议，确定性构建的 {{ kleister_n_docs }} 文档集）上，我们的无凭据规则抽取器开箱即得字段 F1 **{{ kleister_f1_initial }}**，经一轮针对性规则改进后达到 **{{ kleister_f1_improved }}**，两轮的来源归因准确率均为 **{{ kleister_source_acc }}**——每个抽取值都是其引用块中的原文片段。另一项独立测量：将结构化 LLM 输出从客户端 TOOLS 模式函数调用切换到服务端 Schema 约束解码，在完全相同的环境下把 Schema 校验失败从 **{{ tools_schema_failures }}（30%）** 降到 **{{ constrained_schema_failures }}**。本文所有数字均为定点实测（**{{ kleister_measured_date }}**），以数据文件形式提交在仓库的 `content/kleister_results.toml` 中并在构建时注入——**不是**每次站点构建时重新计算——文末给出可复现的命令。

## 我们先攻击了自己的合成数字

我们的 NDA 垂直领域最初随一套作者自写的合成黄金集发布，字段 F1 **{{ synthetic_nda_f1 }}**。这个数字是真实的，但证明的是另一件事：它验证了*流水线正确性*——抽取、归因、指标在新领域端到端全部工作——而不是真实世界的准确率，因为规则正是在被评分的那些文档上调出来的。

于是我们用一个第三方公开基准来攻击它。Kleister-NDA 是一组真实提交给 SEC 的保密协议，标注来自基准作者而非我们。同一个在合成文档上得分 {{ synthetic_nda_f1 }} 的抽取器，在真实文档上只得 **{{ kleister_f1_initial }}**。这个差距——{{ synthetic_nda_f1 }} 对 {{ kleister_f1_initial }}——才是诚实的全貌，也正是不应轻信厂商在自制数据上做的抽取演示的原因，包括我们自己的。

## 分字段结果：两轮对比

Kleister-NDA 只标注四个字段。以下是有标注文档上的字段 F1（"初始" = 在合成集上调好的规则；"改进后" = 针对性修复之后；{{ kleister_n_docs }} 文档集由 train + dev-0 确定性构建——隐藏的 test-A 标注从未使用）：

| 字段 | 有标注文档数 | 初始 F1 | 改进后 F1 |
|---|---|---|---|
| party（缔约方） | 40 | {{ kleister_party_initial }} | **{{ kleister_party_improved }}** |
| effective_date（生效日期） | 29 | {{ kleister_date_initial }} | **{{ kleister_date_improved }}** |
| term（期限） | 14 | {{ kleister_term_initial }} | **{{ kleister_term_improved }}** |
| governing_law（管辖法律） | 40 | {{ kleister_law }} | {{ kleister_law }}（未变） |

综合字段 F1：**{{ kleister_f1_initial }} → {{ kleister_f1_improved }}**。来源归因准确率：两轮均为 **{{ kleister_source_acc }}**。

## 改进具体做了什么

三项针对实测失败模式的定向修复，每项都做了*双侧*规范化（同一个辅助函数同时规范化标注和抽取结果，格式差异永远不会被算作错误），且均有单元测试：

- **party {{ kleister_party_initial }} → {{ kleister_party_improved }}。** 合成集上的启发式规则依赖显式的"Disclosing Party"/"Receiving Party"角色标签——而真实 SEC 保密协议中几乎从不出现。修复：解析开头"by and between …"序言子句的回退规则，并采用有文档记录的"先出现者为披露方"约定（显式标签存在时仍然优先）。
- **effective_date {{ kleister_date_initial }} → {{ kleister_date_improved }}。** 真实文件使用原正则漏掉的法律文书日期格式——"the 6th day of January, 2012"、序数词、日-月-年顺序、独立的信头日期。修复：覆盖这些格式，并加入线索邻近度判定，使靠近"effective"/"entered into"的日期优先于远处的日期。
- **term {{ kleister_term_initial }} → {{ kleister_term_improved }}。** 文字数字时长（"two (2) years"）和"shall terminate"线索。

governing_law 原本已达 {{ kleister_law }}，未作改动。全程有两道回归防线：合成 NDA 评测保持在 {{ synthetic_nda_f1 }}，通用 CUAD 合同基线字节级不变——NDA 规则只*复用*共享辅助函数，没有分叉核心引擎。

## 来源归因准确率 {{ kleister_source_acc }}——源于构造，不是运气

每个抽取字段都携带 `source_block_id`，指标会校验抽取值确实出现在被引用块的文本中（对多值的 party 字段，*每一个*抽取实体都必须出现在引用块中）。规则查找器只会输出其匹配到的那个块中的片段，因此归因在构造上成立——即使字段 F1 还是 {{ kleister_f1_initial }} 时，归因准确率也保持 {{ kleister_source_acc }}。值错但引用诚实是可恢复的失败模式；引用错误的证据则不可恢复。

## 结构化解码：瓶颈是可靠性，不是模型能力

一项独立测量，在 40 文档的 CUAD *合同*集（不是 Kleister）上，两次运行环境完全相同——Lambda A100-40GB、Ollama、`qwen2.5:32b-instruct` @ 32K 上下文：

- **客户端 TOOLS 模式函数调用（instructor）：** **{{ tools_schema_failures }}** 份文档（30%）直接 Schema 校验失败（嵌套 JSON 畸形，例如多包了一层对象），被计为漏检。字段 F1 **{{ mlx_f1 }}**，来源归因准确率 {{ mlx_source_acc }}——部分原因是模型丢掉了引用块 id 的 `#` 前缀。
- **服务端 Schema 约束解码（`response_format=json_schema`）：** Schema 失败 **{{ constrained_schema_failures }}**。字段 F1 **{{ constrained_f1 }}**（两次运行；解码非确定性，±0.01 视为运行噪声），来源归因准确率 {{ constrained_source_acc }}。

相同模型、相同提示词、相同文档。那 30% 的失败类别是*传输层*问题，语法约束解码在传输层将其消除。剩余的归因准确率缺口是 `wrong_span`——值正确但引用了错误的块——任何输出格式约束都无法修复它。

## 诚实的局限

- **Kleister-NDA 只标注 4 个字段**，其中两个标注稀疏（effective_date 29、term 14，共 {{ kleister_n_docs }} 份）。综合 F1 只在可评分范围内取平均。
- **剩余的 party 漏检是结构性的：** 多数是人名或不带公司后缀的标注实体，公司实体正则在设计上就无法输出它们。
- **规则抽取器是确定性、无凭据的下限**，不是能力上限——Kleister 数字没有使用任何 LLM。
- **结构化解码数字来自 CUAD 合同，不是 Kleister**——它衡量的是本地 LLM 路径的输出可靠性，且是另一个语料。
- **这些是定点数字（{{ kleister_measured_date }}）**，以数据形式提交、构建时注入。它们不像我们的清洗基准文章那样实时重算，因为数据集无法提交（见下文），GPU 运行也无法在 CI 中执行。

## 自己复现一下

Kleister-NDA 仓库未声明明确许可证，因此其文件从不提交到本仓库——请将 `KLEISTER_DIR` 指向你自己的检出：

```bash
git clone https://github.com/applicaai/kleister-nda
git clone <repo> && cd contract-rag && uv sync --extra dev
KLEISTER_DIR=../kleister-nda uv run python -m contract_rag.verticals.nda.kleister         # 构建确定性 40 文档集
KLEISTER_DIR=../kleister-nda uv run python -m contract_rag.verticals.nda.kleister --eval  # 字段 F1 + 来源归因准确率
```

数据集构建是确定性的（train + dev-0，带种子的选取），因此在 PDF 解析器的环境差异范围内，你应能复现字段 F1 **{{ kleister_f1_improved }}** / 来源归因准确率 **{{ kleister_source_acc }}**。如果你的数字有实质差异，欢迎提 issue——公开测量的意义正在于此。
