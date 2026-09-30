# v1m4 模型架构与设计思想

> **这份文档写给谁**：想理解 v1m4 **为什么长这样**的人 —— 算法同学、接手维护的工程师、做架构评审的人。
>
> **它不是什么**：不是实现规格。要看逐行的 shape、变量名、初始化器、算子顺序，请读 [`model.md`](model.md)（1808 行的 re-implementation spec）。本文只讲结构、动机和权衡。
>
> **配套**：[`new_optimize.md`](new_optimize.md)（等价性证明与优化方案）、[`optimize.md`](optimize.md)（优化待办）、[`tests/`](tests/)（5 个契约测试 + 参数清点）。

---

## 1. 一句话定位

v1m4 是一个**多场景、多任务的推荐排序模型**：把 133 个用户特征、60 个物品特征、13 条行为序列，压缩成 **8 个 user token + 8 个 item token**，让它们在 **250 个序列 token 构成的记忆库**上互相 attend 两轮，再用 **7 个学习出来的探针**从这 16 个 token 里抽取任务表征，最后经场景硬路由喂给 5 个带门控的任务塔。

它的核心张力来自一个业务事实：

> **一次请求只有 1 个用户，但有 1024 个候选物品。**

整个架构的几乎每一个决策，都是在回答同一个问题：**这块计算应该算 1 次，还是算 1024 次？**

---

## 2. 四条设计主线

### 主线一：一切皆 token

推荐系统的特征天然是异构的 —— 有 4 维的性别 one-hot，有 128 维的 user_id embedding，有 500 步的点击序列，有 33 维的图像向量。传统做法是把它们全部 concat 成一个几千维的大向量喂给 MLP。

v1m4 不这么做。它把每一类特征**投影成统一的 128 维 token**：

```
  user 组 (272 维)      ──┐
  user_short (904 维)   ──┤
  user_long (1656 维)   ──┤
  context (560 维)      ──┼──> 各自一个 [128] 线性投影 ──> 8 个 user token
  scene_onehot (10 维)  ──┤
  click 序列均值 (176)  ──┤
  order 序列均值 (136)  ──┤
  全部拼接 (3714 维)    ──┘   ──> [640] 投影 ──> 切 5 块 ──> chunk 0 是第 8 个 token
```

**为什么值得这么做？** 因为一旦所有东西都是同维度的 token，就能用 attention、token mixing、per-token FFN 这些原本为序列建模设计的机制 —— 它们比 MLP 更擅长表达"哪一部分特征此刻更重要"。这是 RankMixer 一脉的思路。

代价是投影参数量：8 个 user token 的投影就占了全模型 **25%** 的参数（2.85M / 11.44M），其中 `user_global` 那个 3714→640 的投影一个人占 2.38M。

### 主线二：user / item 解耦，粒度分离

这是 v1m4 最重要的工程约束，也是它区别于普通多任务模型的地方。

```
        ┌──────────────── 一次请求 ────────────────┐
        │                                          │
   请求级 (R = 1)                          候选级 (C = 1024)
        │                                          │
   user 特征                                item cache 查表
   5 路行为序列                              SimTier 多模态
   序列编码 / K/V 投影                        fusion query
   user token 投影                                 │
   FIM user 分支                                   │
   TIM user K/V                                    │
        │                                          │
        └──────────► 只在 3 个明确的点汇合 ◄────────┘
                     ① fusion query 的加法
                     ② FIM 的 user_tail 加法
                     ③ TIM 的分区归并
```

**关键不变量：user token 从头到尾看不到任何 item 信息。**

这不是巧合，是**数学前提**。FIM 的 user 分支只 attend 序列记忆（用户自己的行为），`UIHeadMixer.mix_user` 的 gather 索引只在 user token 内部取。正因为 user 侧的输出与 item 无关，它才能只算一次、被 1024 个候选共享。

反过来，item token 是**吸收了 user 信息**的 —— 8 个 item token 里有 4 个直接由 `user_global` 的 chunk 与 item cache 的 chunk 配对生成，另外在 FIM 里每个 item token 的前 64 个通道还要加上 `user_tail`。

所以信息流是**单向的**：

```
   user ──────────────────────► item
     ▲                            │
     └──── 永不反向 ──────────────┘
```

这个单向性是整个 serving 优化的地基。一旦哪天有人让 user token 也 attend item token，请求粒度分离就立刻失效，serving 成本会乘以 1024。

### 主线三：序列是记忆库，不是特征

传统 DIN 类做法：把用户点击序列与目标物品做 attention，pooling 成一个向量，拼进特征。

v1m4 的做法：把 5 条行为序列编码成 **250 个 token 的常驻记忆库**，让 16 个 user/item token 各自去查询。

```
  序列组                     编码后视图              长度
  ─────────────────────────────────────────────────────
  user_click_seq      ──►   click                   50
  user_order_seq      ──►   order                   50
  user_click_long_seq ──►   long_click  (450→50)    50
  user_cart_seq       ──►   atc                     50   ← 专属路由
                       └──► atc_generic (50→25)     25   ← 进通用记忆
  upstream_impression ──►   upstream    (50→25)     25
  ─────────────────────────────────────────────────────
                              合计                  250
```

这 250 个 token 被组织成**两种视图**：

| 视图 | 组成 | 谁在用 |
|---|---|---|
| **generic memory**（200 token） | click 50 + order 50 + long_click 50 + atc_generic 25 + upstream 25 | 8 个 user token 全部；4 个 item base token |
| **dedicated routes**（4 × 50 token） | click / atc / order / long_click 各自的完整视图 | 4 个 fusion query token，**每个只看自己那一条** |

**为什么要双视图？** 因为"这个用户总体上喜欢什么"和"这个用户在这条特定行为线上对当前物品什么反应"是两种不同的信号。让 `click` fusion query 只 attend 点击序列，比让它在 200 个混合 token 里自己学会筛选要高效得多 —— 这是一种**结构化的先验**，用架构代替学习。

代价：序列编码只算一次（跨 2 层 FIM 共享），但每层 FIM 都要把这 250 个 token 重新投影成 K/V（每层 8.19M MAC）。这在训练时是**最大单项开销（52%）**，在推理时因为处于请求级而几乎免费（占请求级 72%，但请求级只占总量 1/236）。

### 主线四：任务和场景都是 query

传统多任务模型：共享底层 + N 个任务塔，任务之间的交互靠 MMoE/PLE 的 gate。

v1m4 的 TIM（Task Interaction Module）用了一个更统一的视角：**把"我想要什么信息"表达成 query，让 attention 去 16 个 token 里取**。

7 个 query 分两类：

```
  ┌─ 任务语义 ─────────────────────┐  ┌─ 场景语义 ──────┐
  │  click   atc   order   common  │  │  dd   pdp   pp  │
  └────────────────────────────────┘  └─────────────────┘
              │                                │
              │                                ▼
              │                        场景硬路由：三选一
              │                    scene_hidden = dd·h_dd
              │                                + pdp·h_pdp
              │                                + pp·h_pp
              ▼                                │
        任务塔的输入 ◄──────────────────────────┘
```

`common` 是跨任务共享的表征，它同时进任务塔输入和 gate 输入。`dd/pdp/pp` 是三个场景（Discover 信息流 / 商品详情页 / 搜索推广位）的专属表征，靠 `scene_onehot` 的 0/1 语义做**硬选择**（不是软加权）。

⚠️ 这个硬路由是 v1m4 里最脆弱的一处：它依赖 `scene_onehot` 保持精确的 0/1。一旦对 scene_onehot 做减均值的归一化，三列就不再互斥，硬选择会退化成任意线性组合，且**不会报错**。

---

## 3. 架构总览

```
                        mcconf_brnew_v4.yaml
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   静态特征组                 13 条序列组              item cache
        │                        │                     (slot 30102)
        │                        ▼                        │
        │              ┌──────────────────┐               │
        │              │ 序列视图编码      │               │
        │              │ 5 条流 → 6 个视图 │               │
        │              │ 250 token        │               │
        │              │ (只算一次,跨层共享)│               │
        │              └────────┬─────────┘               │
        ▼                       │                         ▼
  ┌────────────┐                │                  ┌────────────┐
  │ 8 user tok │                │                  │ 8 item tok │
  │  [R,8,128] │                │                  │  [C,8,128] │
  └─────┬──────┘                │                  └─────┬──────┘
        │                       │                        │
        │    ┌──────────────────┴──────────────────┐     │
        │    │   chunk 1..4 配对 → 4 个 fusion query │◄────┤
        │    └─────────────────────────────────────┘     │
        │                                                │
        └────────────────┬───────────────────────────────┘
                         ▼
        ╔════════════════════════════════════════╗
        ║   FIM 第 1 层                          ║
        ║   user 分支: CA(200) → S-FFN → mix     ║
        ║              → user_tail ──────┐       ║
        ║   item 分支: CA(200+4×50) → S-FFN      ║
        ║              → mix ← ─ ─ ─ ─ ─ ┘       ║
        ║              → M-FFN                   ║
        ╠════════════════════════════════════════╣
        ║   FIM 第 2 层（同结构，独立参数）        ║
        ╚════════════════╤═══════════════════════╝
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   ┌─────────────┐              ┌─────────────┐
   │ rank_global │              │     TIM     │
   │   [C,256]   │              │ 7 task token│
   │ 2 层 MLP    │              │  [C,7,128]  │
   └──────┬──────┘              └──────┬──────┘
          │                            │
          │                     ┌──────┴───────┐
          │                     ▼              ▼
          │              4 个任务 token   3 个场景 token
          │                     │              │ 硬路由
          │                     │         scene_hidden
          │                     │              │
          └──────────┬──────────┴──────────────┘
                     ▼
          ┌────────────────────┐
          │  5 个门控任务塔     │
          │  (grouped GEMM)    │
          └─────────┬──────────┘
                    ▼
      click  atc  order  place_order  ads_order
                    +
              item_cache_norm (监控)
```

---

## 4. 五个关键设计的"为什么"

### 4.1 为什么是 8 + 8 个 token？

不是随便定的。8 = `d_model / num_tokens = 128 / 16 = 8`，这个 **depth=8** 是 `UIHeadMixer` 能工作的前提：它要在 16 个 token 之间做通道级的重排，每个 token 贡献 `128/16 = 8` 个通道。

user 侧的 8 个位置是**按信息类型**分的：3 个时间尺度（静态画像 / 短期 / 长期）+ 上下文 + 场景 + 2 个序列摘要 + 1 个全局压缩。item 侧的 8 个是**按信息来源**分的：4 个内容型（cache / 召回 / 图像 / 标题）+ 4 个交互型（4 条行为线的 fusion query）。

如果哪天要加第 9 个 token，`UIHeadMixer` 的 depth 会变成 `128/18`，不整除 —— 整个混合机制会失效。这是个**硬约束**。

### 4.2 为什么 item cache 是 640 维、切成 5 块？

640 = 5 × 128。这个设计一石三鸟：

```
   item cache (640)  ──reshape──►  [5, 128]
                                     │
             chunk 0 ────────────────┼──► item token 0（内容表征）
             chunk 1 ────────────────┼──► 与 user_global chunk 1 配对 → click fusion query
             chunk 2 ────────────────┼──► 与 user_global chunk 2 配对 → atc   fusion query
             chunk 3 ────────────────┼──► 与 user_global chunk 3 配对 → order fusion query
             chunk 4 ────────────────┴──► 与 user_global chunk 4 配对 → long_click fusion query
```

1. **chunk 0 直接做 item token**，不需要额外投影
2. **chunk 1-4 与 user_global 的 chunk 1-4 一一对应**，天然形成 4 个 U-I 交互通道
3. **user 侧和 item 侧共用同一套切分口径**，这让 fusion query 的语义对齐变得自然

而 `user_global` 之所以要投影到 640 而不是 128，就是为了能切成 5 块与 item cache 对齐。

### 4.3 为什么 FIM 是 `CA → S-FFN → UI mix → M-FFN` 这个顺序？

一层 FIM 做四件事，顺序有讲究：

```
  ① Cross Attention   从序列记忆里取信息       ← 先获取外部信息
  ② S-FFN             per-token 独立加工       ← S = Self/Sequential，token 内部消化
  ③ UI Head Mixer     跨 token 交换通道        ← 无参数的信息路由
  ④ M-FFN             per-token 独立加工       ← M = Mixing，消化交换来的信息
```

**为什么 mixer 夹在两个 FFN 中间？** 因为 mixer 是**无参数的纯置换** —— 它只是把通道搬来搬去，不做任何加工。如果它后面不跟一个 FFN，搬过去的信息就没人处理。所以结构必须是"取信息 → 加工 → 交换 → 再加工"。

**为什么 mixer 无参数？** 这是一个刻意的取舍。带参数的 token mixing（如 attention 或全连接）会增加 `16×16×128` 量级的参数和计算；而固定置换是**零参数、零 MAC、单次 gather**。RankMixer 的核心洞察就是：token 间的信息交换不需要学习，一个精心设计的固定置换加上后面的 per-token FFN 就够了 —— 学习能力由 FFN 提供，路由由置换提供。

置换的具体规则：

```
   d_model = 128 拆成两半 (half_dim = 64)，16 个 token 各贡献 depth = 8 个通道

   mix_user:  输出的每个 half，从所有 8 个 user token 各取 8 个通道
              → 8 target × 2 half × 8 source × 8 channel = 1024 个索引（完整置换）

   mix_item_tail: 只处理后半区
              → 8 target × 8 source × 8 channel = 512 个索引
              item 的前半区留给 user_tail
```

### 4.4 为什么 `user_tail` 是单向的、且只传 64 维？

这是 U/I 信息交换的**唯一通道**（fusion query 是在 FIM 之前构造的，不算 FIM 内部交换）：

```
   user_mixed [R,8,128]
        │
        ├── 前 64 通道 ──► 进 user M-FFN
        └── 后 64 通道 ──► user_tail ──┐
                                       │  (跨粒度：R → C)
   item_hs [C,8,128]                   │
        ├── 前 64 通道 + user_tail ◄───┘
        └── 后 64 通道 + item_tail
```

三个设计细节：

1. **只传后半区（64 维）** —— 前半区留给 item 自己，后半区注入 user。这样每个 item token 的一半通道是"纯物品"、一半是"人物交互"，语义上分工明确。传输量也只有 512 float/请求（tile 到 1024 候选是 2 MB，比传完整 K/V 的 16 MB 小一个量级）。
2. **在 user M-FFN 之前取出** —— 意味着 item 拿到的是"带残差但未经最后一层 FFN"的中间态。这是为了让 EGO 只在这一个加法点对齐 COMMON 行，减少粒度转换的次数。
3. **绝不反向** —— item 的信息不会流回 user。这是主线二的不变量。

### 4.5 为什么 TIM 是双 expert **并行**，而不是 MoE？

```
                    query_seed [1,7,128]   ← 纯参数，与输入无关
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
         expert 0                    expert 1
      (独立的 Wq/Wkv/Wo)          (独立的 Wq/Wkv/Wo)
              │                           │
       同一份 16-token memory      同一份 16-token memory
       一次联合 softmax            一次联合 softmax
              │                           │
           Y_0 [B,7,128]              Y_1 [B,7,128]
              └─────────────┬─────────────┘
                            ▼
                   concat → [B,7,256]
                            │
                 7 个独立 [256,128] 融合矩阵
                            │
                      + query_seed（只加一次）
                            │
                 7 个独立 SwiGLU + 残差
                            │
                       [B,7,128]
```

**没有 gate、没有 router、没有 top-k**。两个 expert 恒定都参与，融合权重是学出来的**常量矩阵**（不依赖输入）。

为什么不用 MoE？因为这里的目标不是"节省计算"（MoE 的卖点），而是**表达力**：让同一个 task query 能从两个不同的子空间读取记忆，再由 per-query 的融合矩阵决定怎么组合。7 个 query 有 7 套独立的融合矩阵，所以 `click` 可以主要依赖 expert 0，`order` 可以偏向 expert 1，或者各自取不同的通道子空间。

数学上，`concat(Y_0, Y_1) @ W_t` 等价于 `Y_0 @ W_t^{(0)} + Y_1 @ W_t^{(1)}` —— 是**矩阵系数的线性组合**，比标量门控 `α·Y_0 + (1-α)·Y_1` 表达力强得多。

另一个关键设计：**query 是纯参数**。`query_seed` 不依赖任何输入特征，所以 query 投影在 batch=1 上完成，与候选数无关。这也让"把 Wq 折进 Wk"这类优化成为可能（见 `new_optimize.md` 第二部分）。

### 4.6 为什么任务塔要双层门控？

```
   tower_input [B, 640/768/896]
        │
   Dense(→128) + LeakyReLU  ──► hidden
        │                          │
        │                     × gate0        gate0 = 2·sigmoid(W·gate_input) ∈ (0,2)
        │                          │
   Dense(→64) + LeakyReLU   ──► hidden
        │                          │
        │                     × gate1        gate1 = 2·sigmoid(W·gate_input) ∈ (0,2)
        │                          │
   Dense(→1) ──► sigmoid ──► prediction

   gate_input = [scene_hidden(128), common_token(128), scene_condition(3)] = 259 维
```

门控值域是 **(0, 2)** 而不是标准的 (0, 1) —— 这意味着门既能**抑制**也能**放大**。标准 sigmoid 门只能衰减信号，而 `2·sigmoid` 让模型可以根据场景动态调整每个隐层的增益。

门控的输入是 `gate_input`，包含场景信息（`scene_hidden` + `scene_condition`）和跨任务共享表征（`common`）。所以这是**场景条件化的计算**：同一个任务在 DD 场景和 PDP 场景下，隐层的增益模式可以完全不同。

5 个塔的参数完全独立，但用 grouped GEMM 一次算完（`[5,B,896] @ [5,896,128]`），代价是把 640/768 宽的输入零填充到 896（约 8.6% 的空算）。

---

## 5. 训练与推理是两套不同的图

这是 v1m4 最容易让人困惑的地方 —— 同一个模型类，在训练和推理时跑的是**结构不同的计算图**。

### 5.1 item cache：teacher-student

```
   训练时                                    推理时
   ────────                                  ────────
   item 组 60 特征 (816 维)                   item cache slot 30102
        │                                          │
   DenseTower(816→640)                             │ 直接查表
        │                                          │
   item_teacher ──┐                                │
                  ├─ replace_gradient ─► item_emb ◄┘
   cache_emb ─────┘        │
   (slot 30102)            └── 梯度只流向 teacher
                             前向取 teacher 的值
                             EGO AssignOptimizer 把
                             teacher 输出写回 slot 30102
```

**动机**：item 侧的 60 个特征（816 维）在推理时如果每次都要重算，1024 个候选就是 1024 次 816→640 的投影。把它们**离线预计算并写进 cache**，在线只查表，是巨大的节省。

`replace_gradient(teacher, cache, teacher)` 的语义是：前向用 teacher 的值，反向把梯度导向 teacher（不流向 cache 查表路径）。这样 teacher 能被正常训练，而训练好的输出通过 assign slot 机制沉淀到 cache 里。

**代价**：训练图里有 522,880 个参数（`brv4_item_cache_teacher`）在推理图里**完全不存在**。这也是"训练参数量 11.44M / 推理参数量 10.91M"这个差异的唯一来源。

### 5.2 三处训推分叉

| 模块 | 训练 | 推理 | 为什么 |
|---|---|---|---|
| **FIM item 分支的 attention** | `masked_attention`（同 batch） | `request_shared_attention`（记忆保持请求级） | 推理时记忆是 R 行、query 是 C·R 行，需要显式对齐 `[C,R]` 行序 |
| **TIM attention** | 分离 logits → concat → 一次联合 softmax | 分区 online-softmax 归并 | 推理时让 user 侧的 K/V、QK、AV 全部留在 R=1，只把归约后的统计量广播到候选级 |
| **SimTier 直方图** | `unsorted_segment_sum`（线性时间） | 8-bin 分块 Equal | `unsorted_segment_sum` 经 tf2onnx 会下降成 Unique 节点，**TRT 无法解析** |
| **编译入口** | `ego.compile` | `compile_with_request_attention` | 后者 monkey-patch `GraphEditor.fit_batch_size`，阻止它在已注册 scope 内重新 tile |

⚠️ TIM 的两条路径**数学上精确等价**（log-sum-exp 分区归并恒等式），实测 fp64 下差 7.3e-16、fp32 下差 4.9e-07。详见 `new_optimize.md` 第一部分与 `tests/tim_path_contracts.py`。

### 5.3 归一化的训推一致性

模型用 `ego.DenseTower(norms=[True])`，它内部是 `INormalization`：

- **训练和推理的前向都用 moving stats**（`call` 里没有 mode 分支）
- batch 统计只用于 EMA 更新，不参与前向
- 默认 `use_beta_gamma=False`，即纯标准化，无可学习仿射
- **会减均值**

⚠️ 这和标准 Keras `BatchNormalization`（训练用 batch stats、推理用 moving stats）**不一样**。EGO 另有一个 `IBatchNormalization` 才有 mode 分支。搞混这两个会导致训推一致性判断出错。

---

## 6. 规模画像

### 6.1 参数分布（总 11,437,381 可训练 + 20,744 moving stats）

```
  FIM × 2 层        4,637,184  40.54% |##############
  user token 投影   2,853,888  24.95% |########
  TIM               1,895,680  16.57% |######
  任务塔 × 5          795,333   6.95% |##
  item token          750,336   6.56% |##
  rank_global         393,984   3.44% |#
  序列流编码           110,976   0.97% |
```

两个观察：

- **FIM 独占 40%**。每层 2.32M，其中两个 SwiGLU（S-FFN + M-FFN）就占 1.59M（68%）—— 因为它们是 per-token 独立的：16 个 token × (128×256 + 128×128)。
- **序列流编码只有 1%**，但它承载了 250 个 token 的记忆库。参数少是因为投影是**跨 token 共享**的（一条流一个 `[in_dim, 128]`），而 FIM/TIM 的投影是 **per-token 独立**的。

### 6.2 计算分布：训练与推理是**两个完全不同的画像**

**推理**（1 请求 + 1024 候选，候选级 5.27M MAC/候选 = 5.40G MAC/请求；请求级仅 22.9M，**比 1:236**）：

```
  FIM item 分支 ×2   2,609,152  49.51% |#################
  TIM                1,384,448  26.27% |#########
  任务塔               842,560  15.99% |#####
  rank_global          229,376   4.35% |#
  item token 投影      161,024   3.06% |#
  SimTier cosine        42,900   0.81% |
```

**训练**（每样本 31.2M MAC，所有特征都是 per-sample）：

```
  FIM route K/V ×2  16,384,000  52.48% |##################
  序列流投影          3,635,200  11.64% |####
  FIM user 分支 ×2   2,916,352   9.34% |###
  user token 投影    2,852,352   9.14% |###
  FIM item 分支 ×2   2,609,152   8.36% |###
  TIM                1,384,448   4.43% |##
  任务塔               842,560   2.70% |#
  rank_global          393,216   1.26% |
  item token 投影      161,024   0.52% |
  SimTier cosine        42,900   0.14% |
```

**这个反差是理解 v1m4 性能特性的关键**：

- 推理时瓶颈在**候选级**（FIM item 分支 + TIM + 任务塔 = 92%），优化目标是减少每候选的 MAC 与显存
- 训练时瓶颈在**序列侧**（route K/V + 序列流投影 = 64%），因为这些在推理时是请求级（÷1024），在训练时却是 per-sample
- 所以**同一个优化在训练和推理的收益可能差三个数量级**。`optimize.md` 里每一项都标注了收益归属，原因就在这里

---

## 7. 思想谱系

v1m4 不是凭空设计的，它是几条研究/工程线的合成：

| 来源 | 借鉴了什么 | 在 v1m4 里的体现 |
|---|---|---|
| **RankMixer** | token grouping + per-token FFN + 无参数 token mixing | 8+8 token 布局、`GroupedSwiGLU`、`UIHeadMixer` 的常量索引 gather |
| **Uniformer**（本仓 `cart_rr_id_uniformer`） | grouped 投影、request-shared attention、UI 分离的 M-FFN、chunked cache、层级直方图 | `fim_block.py` 几乎整体照此重写；`request_attention.py`、`simtier_v1m4.py` 直接源自它 |
| **Transformer** | cross-attention、PreLN 残差结构、SwiGLU FFN、multi-head | FIM 的 `CA → +残差 → FFN → +残差` 四段式；TIM 的 joint softmax |
| **DIN / SimTier** | 目标物品与行为序列的相似度特征 | 6 路短序列 + 2 路长序列的多粒度余弦直方图（231/231/204 维） |
| **FlashAttention** | online softmax 的分区归并 | TIM serving 路径的 `(max, den, num)` 三元组合并 |
| **ESMM / MMoE 家族** | 多任务塔、任务间表征共享 | 5 个独立塔 + `common` token 共享 + `order` 类任务复用表征 |
| **知识蒸馏 / cache 预计算** | 把重计算从在线挪到离线 | item cache 的 teacher-student + assign slot |

**v1m4 自己的贡献**主要是把这些拼成一个**在 1:1024 极端不对称下仍然高效**的结构，特别是：

- 用 `user_global` 的 640 维 chunked 输出，让 user 侧信息以**结构化配对**的方式注入 4 个 item token（而不是简单 concat）
- 用 dedicated/generic 双视图，把"某条行为线的专属交互"和"全局行为上下文"分开
- 用 TIM 的 7 query（4 任务 + 3 场景）统一表达任务与场景，再用硬路由选场景

---

## 8. 已知妥协与权衡

诚实地列出模型里**不完美但有意保留**的部分：

| 妥协 | 现状 | 为什么保留 |
|---|---|---|
| **`brv4_scene_token` 加了 `input_norm=True`** | scene_onehot 的 token 表征被减了均值，失去 0/1 语义 | 硬路由读的是**原始** `scene_onehot_input`，所以功能没坏；但作为一个 token，它的信息量可能下降 |
| **`click_mean` / `order_mean` 加了 `input_norm=True`** | 序列均值的分布高度依赖有效长度，减均值会把"无行为"用户映射到非零常量 | 同上，功能正常但语义上不如 RMSNorm 干净（RMSNorm 不减均值，0 仍是 0） |
| **三个 order 类任务的 `tower_input` 内容相同但算了 3 次** | `tf.concat` 因 `name` 不同不会被 CSE 合并 | 保留独立 name 便于图检查；去重是 `optimize.md` S2 的待办 |
| **任务塔零填充到 896** | click(640)/atc(768) 被填充，约 8.6% 空算 | 换来一个 grouped GEMM 而非 3 个小 matmul；净收益未确认（`optimize.md` S7） |
| **`atc_generic` 在投影后池化，`upstream` 在投影前池化** | 两者不对称：前者的 position embedding 是相邻两行的平均 | `atc` 必须保留 native 50 视图作 dedicated route，所以只能先投影 |
| **`type_embedding * (type_index+1)`** | 5 条流的 type 偏移初始幅值差 5 倍 | per-stream 已是独立变量，乘子不省参数；改动会破坏 checkpoint |
| **每层 FIM 重新投影 250 个序列 token 的 K/V** | 训练时占 52% 计算 | per-layer K/V 是架构选择；跨层共享会改变模型表达力 |
| **K/V 投影去掉了 bias** | 失去了一个可学习的 attention logit prior | 换取参数精简；Q 侧去 bias 是无损的（per-query Wq 已能张成任意向量） |

---

## 9. 快速上手路径

如果你要接手这个模型，建议按这个顺序读：

1. **本文档 §2（四条主线）** —— 建立整体直觉，15 分钟
2. **`model.md` §4（端到端数据流）** —— 看清每一步的 shape 和 batch 粒度
3. **`models/rankmixer_tim_model_v1m4.py` 的 `build_fim_inputs` 与 `build_esmm`** —— 主编排，约 100 行
4. **`models/fim_block.py` 的 `call`** —— FIM 单层的完整顺序，约 50 行
5. **`models/tim_block.py` 的 `_forward`** —— TIM 的完整流程，约 65 行
6. **`tests/tim_numpy_contracts.py`** —— 纯 NumPy，不需要 EGO 环境，`python3` 直接跑，能最快建立对数值行为的信心

需要改代码前必读：

- **`model.md` §8** —— 24 条建图期不变量
- **`model.md` §9.2** —— 10 个已知陷阱（尤其 XLA rank-3 kernel、SimTier 分箱、`unsorted_segment_sum` 的 TRT 问题）
- **`model.md` §10** —— 验收清单

需要动性能前必读：

- **`optimize.md`** —— A–H 待办（A–F、H 已落地，G 已排除）
- **`new_optimize.md`** —— TIM 两路径等价性证明 + QK 折叠方案

---

## 附：术语表

| 缩写 | 全称 | 含义 |
|---|---|---|
| FIM | Feature Interaction Module | 2 层，每层做 CA → S-FFN → UI mix → M-FFN |
| TIM | Task Interaction Module | 双 expert 并行 cross-attention，产出 7 个 task token |
| S-FFN | Self/Sequential FFN | attention 后的 per-token SwiGLU |
| M-FFN | Mixing FFN | token 混合后的 per-token SwiGLU |
| CA | Cross Attention | token 作为 query，序列记忆作为 K/V |
| generic memory | — | 200 token 的通用记忆（5 路拼接） |
| dedicated route | — | 4 条专属路由，各 50 token，供 4 个 fusion query 独享 |
| fusion query | — | item token 4..7，由 user_global chunk 与 item cache chunk 配对生成 |
| user_tail | — | user_mixed 的后 64 通道，FIM 内 U→I 的唯一通道 |
| SimTier | Similarity Tier | 目标物品与行为序列的余弦相似度多粒度直方图特征 |
| assign slot | — | EGO 机制：训练时把 tensor 写入指定 slot，推理时查表 |
| COMMON / ITEM | — | EGO 的 feature_type：请求级 / 候选级 |
| R / C | — | 请求数 / 每请求候选数，参考 R=1、C=1024 |
