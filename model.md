# model_v1m4 实现规格（re-implementation plan）

> **本文档的用途**：作为一份**可执行的实现规格**。另一个 AI 或工程师只读本文档（不读源码），应当能重新写出功能、形状、变量命名、数值语义完全一致的 `model_v1m4`。
>
> **被描述的对象**：`ego_tasks/br_rr_unify/model/model_v1m4.py` 及其依赖链，特征基座 `configs/mcconf_brnew_v4.yaml`。
>
> **写作原则**：所有维度、shape、变量名、初始化器、算子顺序都来自源码实测，不是推测。凡是"必须如此否则不等价"的地方都标了 ⚠️。凡是刻意的设计选择（而非偶然）都标了 💡 并给出理由。
>
> **配套文档**：`new_optimize.md`（TIM 两条 attention 路径的等价性证明 + QK 折叠方案）、`optimize.md`（效率优化待办）、`FIM.md`、`TIM.md`、`plan_v1m4_brnew_v4.md`（历史规划）。

---

## 0. 一句话概括

一个 **user/item 解耦** 的推荐排序模型：把 user 侧特征压成 **8 个 128 维 token**、item 侧压成 **8 个 128 维 token**，两侧各自与 **6 路用户行为序列视图**（共 250 个 token）做 **2 层 FIM**（cross-attention → per-token SwiGLU → U/I 通道混合 → per-token SwiGLU），再把 16 个 token 作为 memory 喂给 **双 expert 并行 TIM**（7 个 learned query 做一次 16 位联合 softmax 的 cross-attention），产出 7 个 task token，最后经场景硬路由 + 5 个带门控的任务塔输出 5 个 sigmoid 预测。

**核心工程约束**：user 侧全程保持**请求粒度**（serving 时 batch=R=1），item 侧是**候选粒度**（batch=C·R，参考 C=1024）。两者只在明确标注的"对齐点"汇合，绝不提前 tile。

---

## 1. 运行环境与文件布局

### 1.1 环境

| 项 | 值 |
|---|---|
| 框架 | TensorFlow，**TF1 图模式**：`models/base_model.py` 头部 `import tensorflow.compat.v1 as tf` + `tf.compat.v1.disable_eager_execution()` |
| 平台 API | EGO（`import ego`；`ego.DenseTower`、`ego.replace_gradient`、`ego.get_slots`、`ego.get_dense_feature`、`ego.Target`、`ego.OfflineRound`、`ego.OnlineRound`、`ego.compile`） |
| 硬件 | NVIDIA A30 |
| 训练 | XLA |
| 推理 | TensorRT（TRT） |
| serving batch | 1 请求 = R=1 user（COMMON）+ C 候选（ITEM），参考 C=1024 |
| training batch | 每个样本一个 user-item 对，所有特征 batch = `B_train` |

⚠️ **`tim_block.py` / `fim_block.py` 用顶层 `tf.variable_scope` 与 `tf.get_variable`**（不是 `tf.compat.v1.` 前缀），这要求运行时的 `tensorflow` 顶层就暴露 v1 API。`rankmixer_v2_layers.py` 与 `base_model.py` 则显式用 `tensorflow.compat.v1`。两种写法在同一运行时共存，重写时**照抄各自文件的写法**，不要统一。

### 1.2 文件与职责

```
model_v1m4.py                          入口：超参、extra slots、构图、export.yaml、Round 定义与编译
models/
  base_model.py                        BaseModel：读特征（get_slots / get_dense_feature）、train/serve 分支、变量与打印
  rankmixer_tim_model_v1m4.py          RankMixerTIMModelV1M4：全部特征→token→FIM→TIM→塔→loss 的编排
  fim_block.py                         FIMBlock（Keras Layer）+ UIHeadMixer + grouped_linear/grouped_glorot/rms_norm/GroupedSwiGLU
  tim_block.py                         ParallelTIMBlock（普通 object，TF1 variable_scope 风格）
  rankmixer_v2_layers.py               tim_block 的底层 helper：_independent_kernels / _token_independent_matmul
                                       / apply_rms_norm / packed_per_token_swiglu（FIM 不用这个文件）
utils/
  request_attention.py                 masked_attention / request_shared_attention / request_shared_tim_attention
                                       / compile_with_request_attention
  simtier_v1m4.py                      request_cosines / _histogram_counts / _coarse_from_fine
                                       / short_simtier / long_simtier
  fg_tool.py                           FGConf, ModelColumnUtil：解析 mcconf → col_slots/col_dims/col_aggregators
  tools.py                             convert_pooling 等
configs/mcconf_brnew_v4.yaml           特征基座（分组、slot_id、dim、aggregator、length）
tests/                                 tf_fim_contracts.py / compiled_fim_checks.py / tim_numpy_contracts.py
                                       / tim_path_contracts.py / tim_qk_fold_contracts.py
```

💡 **依赖方向的关键事实**：`fim_block.py` **不** import `rankmixer_v2_layers`（它自带 `grouped_linear`/`rms_norm`/`GroupedSwiGLU`/`grouped_glorot`）；只有 `tim_block.py` 依赖 `rankmixer_v2_layers`。而 `rankmixer_v2_layers` 顶层 `import ego`，所以 **import `tim_block` 就需要 EGO 环境**，import `fim_block` 不需要。这决定了哪些测试能在无 EGO 环境跑。

---

## 2. 特征输入契约

### 2.1 mcconf 静态特征组

`configs/mcconf_brnew_v4.yaml` 的顶层 key 就是组名，`self.groups = list(self.mcconf.keys())`。每个特征条目形如：

```yaml
- {name: user_id, slot_id: 30000, dim: 128, aggregator: avg, is_shared: true}
```

| 组名 | 特征数 | concat 后维度 | 用途 |
|---|---:|---:|---|
| `user` | 13 | **272** | user token 0 |
| `user_short` | 46 | **904** | user token 1 |
| `user_long` | 74 | **1656** | user token 2 |
| `context` | 24 | **432** | user token 3（与 indicator 拼接） |
| `indicator_ctx_fea` | 1 | **128** | user token 3（`context_bundle_1`, slot 63122） |
| `item` | 60 | **816** | 训练态 item cache teacher 的输入 |
| `recall` | 5 | **80** | item token 1 |
| `target_item_query_img` | 1 | **33** | SimTier 的 image query |
| `target_item_query_title` | 1 | **33** | SimTier 的 title query |
| `item_cache_slot` | 1 | slot **30102**, dim **640** | assign cache（被 `pop` 出 mcconf，单独处理） |

**`context` token 的输入 = 432 + 128 = 560。**
**`user_global` token 的输入 = 272+904+1656 (=2832) + 560 + 10 (scene_onehot) + 176 (click_mean) + 136 (order_mean) = 3714。**

⚠️ `user` / `user_short` / `user_long` 的划分依据是**有效时间窗**（静态=画像/身份/设备；短期=≤3d 行为+1d 快照+最近 session+购物车/订单状态快照；长期=7d 及以上+21d 分段+全序列聚合），**不是特征名前缀**。实测 `user_short` 里有 24 个 `lt_` 前缀、16 个 `st_` 前缀；`user_long` 里有 49 个 `lt_`、15 个 `st_`、8 个含 `recent`。前缀与时间尺度不对应，不要按前缀重新分组。

### 2.2 序列组（全部 `aggregator: tile_nf`）

| 组名 | 特征数 | `length` | 每步 concat 维度 |
|---|---:|---:|---:|
| `user_click_seq` | 11 | 50 | **176** |
| `user_order_seq` | 7 | 50 | **136** |
| `user_cart_seq` | 5 | 50 | **112** |
| `upstream_impression_seq` | 6 | 50 | **128** |
| `user_click_long_seq` | 2 | **500** | **80** |
| `ub_clk_img_seq` / `ub_cart_img_seq` / `ub_order_img_seq` | 各 1 | 50 | **33** |
| `ub_clk_title_seq` / `ub_cart_title_seq` / `ub_order_title_seq` | 各 1 | 50 | **33** |
| `ub_clk_500_img_seq` / `ub_order_500_img_seq` | 各 1 | **500** | **33** |

`length` 字段决定静态 padded 长度：`utils/fg_tool.py` 对 tile/tile_nf 特征把 dim 写成二元组 `(length, dim)`，缺 `length` 直接 `AttributeError`；再传给 `ego.get_slots(dims=[(length,dim)], poolings=[TILE_NF])`，返回 **`(values, lengths)` 元组**，`values` 形状 `[B, length, dim]`。

### 2.3 dense 特征与 assign cache

| 名称 | 来源 | dim | feature_type |
|---|---|---:|---|
| `scene_onehot` | `ego.get_dense_feature` | **10** | `COMMON` |
| `assign_slot_emb` | `ego.get_slots(slots=[30102], dims=[640], poolings=[SUM], is_assign_slot=True)` | **640** | `ITEM` |

`dense_dim=0`、`onehot_dim=0`（构造函数入参），所以 `BaseModel.build_inputs` 里那两个 `get_dense_feature("dense"/"onehot")` 分支**不启用**。
`register_hanging_slot=False` ⇒ `assert len(hanging_ids) == 0`。

### 2.4 入口文件必须做的声明

```python
ego.import_extra_slots(
    [2001, 2002, 2010, 2011, 3001, 8000, 8001, 12001, 12002, 12003, 12004, 12005],
    [64, 16, 16, 16, 16, 33, 33, 64, 16, 16, 16, 8],
)
ego.config_feature_admission_evict(slots=[30102], delete_threshold=0,
                                   delete_after_unseen_days=9999)
```

💡 `import_extra_slots` 的这些 sparse hash slot 是**特征基座引用到、但本身不是 mcconf slot_id** 的槽位，serving 导出时必须声明。列表由 `configs/tools/gen_extra_slots.py` 从 `mcconf_brnew_v4.yaml` 生成。
💡 `config_feature_admission_evict` 对 assign cache slot 30102 关闭淘汰（`delete_threshold=0` + `unseen_days=9999`），否则 item cache 会被特征准入策略清掉。

入口还要校验 mcconf：

```python
item_cache_slot = mcyaml.pop("item_cache_slot", None)      # ⚠️ 必须 pop，不能留在 mcconf 里
assert item_cache_slot and item_cache_slot[0]["slot_id"] == 30102
assert item_cache_slot[0]["dim"] == 640
```

### 2.5 export.yaml

```python
export_conf = model.mcutil.to_export_yaml(onehot_slots=[], dense_slots=[])
# 确保 sparse 导出里含 30102（不在则 append）
export_conf["dense"] = [{"export_name": "scene_onehot", "slot_ids": [53903]}]
yaml.safe_dump(export_conf, open("configs/export.yaml", "w"))
```

---

## 3. 全局超参与类常量

### 3.1 入口超参（`model_v1m4.py`）

```python
FIM_LAYERS     = 2
FIM_HEADS      = 4
FIM_HIDDEN_DIM = 128
TIM_HEADS      = 8
TIM_HIDDEN_DIM = 128
```

构造：

```python
model = RankMixerTIMModelV1M4(
    None, mcyaml,
    clipnorm=True, register_hanging_slot=False,
    dense_dim=0, onehot_dim=0,
    fim_layers=FIM_LAYERS, fim_heads=FIM_HEADS, fim_hidden_dim=FIM_HIDDEN_DIM,
    tim_heads=TIM_HEADS, tim_hidden_dim=TIM_HIDDEN_DIM,
)
```

### 3.2 模型内常量（`RankMixerTIMModelV1M4`）

```python
USER_TOKEN_GROUPS = ("user", "user_short", "user_long")
CONTEXT_GROUPS    = ("context", "indicator_ctx_fea")
ITEM_GROUPS       = ("item",)
RECALL_GROUPS     = ("recall",)
QUERY_GROUPS      = ("target_item_query_img", "target_item_query_title")
IMAGE_SEQUENCE_GROUPS = ("ub_clk_img_seq", "ub_cart_img_seq", "ub_order_img_seq")
TITLE_SEQUENCE_GROUPS = ("ub_clk_title_seq", "ub_cart_title_seq", "ub_order_title_seq")
LONG_IMAGE_SEQUENCE_GROUPS = ("ub_clk_500_img_seq", "ub_order_500_img_seq")

# route -> (源组, 切片起点, 切片终点, 编码后 stream token 数, generic token 数)
SEQUENCE_SPECS = (
    ("click",      "user_click_seq",          0,  50,  50, 50),
    ("order",      "user_order_seq",          0,  50,  50, 50),
    ("long_click", "user_click_long_seq",    50, 500,  50, 50),
    ("atc",        "user_cart_seq",           0,  50,  50, 25),
    ("upstream",   "upstream_impression_seq", 0,  50,  25, 25),
)
ITEM_TOKEN_COUNT = 8
USER_TOKEN_COUNT = 8
SIMTIER_SHORT_DIMS = 231        # 3 route x (50 + 6 + 21) = 3 x 77
SIMTIER_LONG_DIM   = 204        # 2 route x (21 + 81)     = 2 x 102

TASK_NAMES      = ("click", "atc", "order", "place_order", "ads_order")
TIM_TOKEN_NAMES = ("click", "atc", "order", "dd", "pdp", "pp", "common")   # 7
TASK_TOWER_INPUT_DIM = 896
TASK_GATE_INPUT_DIM  = 260

# __init__ 内
self.param_initializer = tf.glorot_uniform_initializer()   # ⚠️ 覆盖 BaseModel 的 truncated_normal(0.001)
self.param_regularizer = None                              # 继承自 BaseModel，必须为 None
self.assign_slot_id  = [30102]
self.token_dim       = 128
self.tower_dim       = 640
self.cache_chunks    = 5          # 640 / 128
self.num_heads       = fim_heads      # 4
self.swiglu_hidden_dim = fim_hidden_dim  # 128
self.scene_onehot_dim = 10
self._phases = [{"name": "p1"}]
```

⚠️ `TASK_NAMES` 与 `TIM_TOKEN_NAMES` **刻意不互相派生**：TIM 的 7 个 query 里 `dd/pdp/pp/common` 不是预测任务，而 `place_order/ads_order` 没有各自的 TIM token（复用 `order`）。

### 3.3 FIM 模块常量（`fim_block.py`）

```python
USER_TOKEN_COUNT, ITEM_TOKEN_COUNT, ITEM_BASE_TOKEN_COUNT = 8, 8, 4
GENERIC_MEMORY = (("click",50), ("order",50), ("long_click",50),
                  ("atc_generic",25), ("upstream",25))          # 合计 200
GENERIC_ROUTES = ("click","order","long_click","atc_generic","upstream")
GENERIC_TOKEN_COUNT = 200
ROUTE_LENGTHS = {"click":50, "order":50, "long_click":50,
                 "atc":50, "atc_generic":25, "upstream":25}     # 6 路，合计 250
SEQUENCE_ROUTES = tuple(ROUTE_LENGTHS)
FUSION_ROUTES = ("click", "atc", "order", "long_click")         # item token 4..7 的专属路由
```

⚠️ **`GENERIC_MEMORY` 与 `ROUTE_LENGTHS` 的语义差别**：`atc`（native 50）只作为 **dedicated route** 存在，进 generic memory 的是池化后的 `atc_generic`（25）。`upstream` 没有 dedicated route，它的 `stream_tokens` 就等于 `generic_tokens`。

### 3.4 `_check_sequence_specs` 的四条建图期断言

```python
1. route 名不重复
2. stream_tokens <= end - start                       # 不能凭空造 token
3. (end - start) % stream_tokens == 0                 # _pool_fixed 要求整除
4. generic_tokens <= stream_tokens 且 stream_tokens % generic_tokens == 0
5. set(FUSION_ROUTES) ⊆ set(routes)                   # 每个 fusion route 必须有编码流
```

实测：450%50=0、50%25=0、50%50=0 全部满足。⚠️ 断言 3 是 `_pool_fixed` 的前提，**改 `SEQUENCE_SPECS` 时必须同步检查**。

---

## 4. 端到端数据流

```
mcconf_brnew_v4.yaml
   │
   ├─ build_inputs()      → sparse_feature_map[name] = tensor 或 (values, lengths)
   │                        scene_onehot_input [B,10] (clip 0..1)
   ├─ build_group_input() → group_inputs[静态组] = concat(-1)
   │                        sequence_group_inputs[seq组] = concat(-1)  [B,L,Σdim]
   │                        sequence_lengths[seq组]      = reduce_max(feature[1])  [B]
   │                        assign_slot_embs [B,640]
   │
   ├─ _build_sequence_tokens()  → 6 个视图 {(tensor[B,L,128], mask[B,L])}
   │      click 50 / order 50 / long_click 50 / atc 50 / atc_generic 25 / upstream 25
   │
   ├─ _build_user_tokens()      → user_tokens [B_u, 8, 128]
   │                              user_global_chunks [B_u, 5, 128]
   ├─ _build_item_embedding()   → item_emb [B_i, 640], item_cache_norm [B_i, 1]
   ├─ _build_item_tokens()      → item_tokens [B_i, 8, 128]
   │        （内部用 user_global_chunks[:,1:5,:] 与 item_chunks[:,1:5,:] 造 4 个 fusion query）
   │
   ├─ for layer in 1..2:  FIMBlock(layer).call(user_tokens, item_tokens, sequence_tokens,
   │                                           training=is_training_mode())
   │        → user_tokens [B_u,8,128], item_tokens [B_i,8,128]
   │
   ├─ _build_rank_global(user_tokens, item_tokens)  → [B_i, 256]
   ├─ _build_tim(user_tokens, item_tokens, training) → tim_tokens [B_i, 7, 128]
   │
   ├─ hidden = dict(zip(TIM_TOKEN_NAMES, unstack(tim_tokens, axis=1)))   # 7 × [B_i,128]
   ├─ scene_condition [B,3] = [dd, pdp, pp]        ← 原始 scene_onehot 的硬路由
   ├─ scene_hidden [B,128] = dd·h_dd + pdp·h_pdp + pp·h_pp
   ├─ gate_input [B,259] = concat(scene_hidden, hidden["common"], scene_condition)
   ├─ tower_inputs[t] = concat(rank_global, hidden[routes[t]]..., scene_hidden)
   │        click 640 / atc 768 / order|place_order|ads_order 896
   ├─ _task_towers(...) → grouped_predictions [B,5,1] → 5 × sigmoid
   │
   └─ loss = -Σ_t reduce_sum(BCE(label_t, pred_t) * weight_t)
```

**batch 语义**：`B_u` = 请求行数（serving R，训练 `B_train`）；`B_i` = 候选行数（serving C·R，训练 `B_train`）。训练态两者相等。

---

## 5. 分模块规格

### 5.1 `build_inputs()`

```python
super().build_inputs()                       # BaseModel 读 sparse_input
assert len(self.used_slot_map) == len(self.sparse_input)
self.sparse_feature_map = {list(self.used_slot_map.values())[i]: t
                           for i, t in enumerate(self.sparse_input)}
# back_propagation: False 的特征冻结梯度
for conf in chain(*self.mcconf.values()):
    if conf.get("back_propagation", True): continue
    f = self.sparse_feature_map[conf["name"]]
    if conf["aggregator"] in ("tile", "tile_nf"):
        self.sparse_feature_map[conf["name"]] = (tf.stop_gradient(f[0]), f[1])
    else:
        self.sparse_feature_map[conf["name"]] = tf.stop_gradient(f)
self.scene_onehot_input = tf.clip_by_value(
    ego.get_dense_feature(name="scene_onehot", dim=10,
                          feature_type=ego.FeatureType.COMMON),
    0.0, 1.0, name="scene_onehot_clipped")
```

⚠️ **`tf.stop_gradient` 而非自定义 `ignore_grad`**：历史上曾用 `utils/ops.ignore_grad`，现已换成 `tf.stop_gradient`（因此 `utils/ops.py` 已成孤儿文件）。
⚠️ **`scene_onehot` 只 clip，绝不归一化**：`_scene_condition` 依赖 `onehot[:,8:9]`、`onehot[:,9:10]` 的 0/1 语义做硬路由，减均值会让硬选择退化成任意线性组合。

`BaseModel.build_inputs` 的 tile/非 tile 分流：tile/tile_nf 的 slot **逐个**调 `ego.get_slots(name=str(slot_id), slots=[slot_id], dims=[dim], poolings=[TILE_NF], feature_type=ITEM)`；非 tile 的**批量**调一次再 `tf.split`。最后按 `slot_id` 排序对齐回 `used_slot_map` 的顺序。serving 分支还按 `mcutil.get_shared_names()` 把特征分成 shared（`FeatureType.COMMON`）与 non-shared（`FeatureType.UNDEFINED`）两路。

### 5.2 `build_group_input()`

```python
self._check_feature_groups()          # 必需组缺失则 KeyError
for group_name in self.groups:
    features = [self.sparse_feature_map[c["name"]] for c in self.mcconf[group_name]]
    if not group_name.endswith("seq"):
        self.group_inputs[group_name] = tf.concat(features, axis=-1);  continue
    # ⚠️ 序列组必须全是 tile_nf 元组
    if not all(isinstance(f, tuple) for f in features):
        raise ValueError("sequence group {} mixes static features".format(group_name))
    lengths = [f[0].get_shape().as_list()[1] for f in features]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("unaligned sequence features in {}".format(group_name))
    self.sequence_group_inputs[group_name] = tf.concat([f[0] for f in features], axis=-1)
    self.sequence_lengths[group_name] = tf.reduce_max(
        tf.concat([f[1] for f in features], axis=-1), axis=-1)
assign_slot = ego.get_slots(name="assign_slot_emb", slots=[30102], dims=[640],
                            poolings=[ego.Pooling.SUM], is_assign_slot=True,
                            feature_type=ego.FeatureType.ITEM)
self.assign_slot_embs = tf.concat(assign_slot, axis=-1)
```

💡 **组名以 `seq` 结尾是判定序列组的唯一依据**。两道 `raise` 是刻意的防线：EGO 对序列组里的静态特征会**静默丢弃**，而长度不齐会导致 concat 错位。
💡 `sequence_lengths` 取组内各特征真实长度的 **max**（不是 min），因为同组不同特征可能因 hash 命中情况长度略有差异，取 max 保证不漏数据。这是**组级一个长度**，组内特征共享。

### 5.3 两个基础原语

**`_mask(lengths, max_len)`** —— 静态方法：

```python
lengths = tf.cast(lengths, tf.int32)
if lengths.get_shape().ndims == 2: lengths = tf.squeeze(lengths, axis=-1)
return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)   # [B, max_len]
```

**`_project_token(tensor, name, hidden_dims=None, input_norm=False)`** —— 所有 token 投影的唯一入口：

```python
hidden_dims = hidden_dims or [self.token_dim]        # 默认 [128]，单层
return ego.DenseTower(
    name=name,
    output_dims=hidden_dims,
    kernel_initializers=[tf.glorot_uniform_initializer()] * len(hidden_dims),
    bias_initializers=[tf.zeros_initializer()] * len(hidden_dims),
    activations=[None] * len(hidden_dims),           # ⚠️ 纯线性，无激活
    norms=[input_norm] + [False] * (len(hidden_dims) - 1),
    use_bias=True,
)(tensor)
```

⚠️ **`activations=[None]`、`norms` 只有第 0 层可能为 True**：`input_norm=True` 时 `DenseTower` 会插入 `Normalization(name="{name}/input_norm")`（即 `INormalization`），它 `tf.nn.moments(axes=除最后一维外全部)` 统计、**训练与推理前向都用 moving stats**（`call` 无 mode 分支）、默认 `use_beta_gamma=False`（纯标准化，无可学习仿射）、**会减均值**。
⚠️ `DenseTower` 的变量命名是 `{name}/layer_{i}/{kernel,bias}`（EGO `layer.py` 用 `K.Dense(name=self._name + "/layer_{idx+1}")`），`norms=[True,...]` 时另有 `{name}/input_norm`，`norms[i]=True (i≥1)` 时另有 `{name}/norm_layer{i}`。
💡 **归一化职责在 `_project_token` 内部**，不再在外部对特征组单独调 `ego.Normalization`（历史上曾有 `_normalize` 方法，已删除）。

**`input_norm=True` 的调用点（10 处）**：`brv4_user_token`、`brv4_user_short_token`、`brv4_user_long_token`、`brv4_context_token`、`brv4_scene_token`、`brv4_click_seq_seed`、`brv4_order_seq_seed`、`brv4_user_global_token`、`brv4_recall_token`、`brv4_item_cache_teacher`（`norms=[True]`）、`brv4_rank_user_a`、`brv4_rank_item_b`。
**`input_norm=False`（默认）的调用点**：`brv4_img_token`、`brv4_title_token` —— 因为它们的输入是 SimTier 的余弦/直方图特征，值域已经受控。

### 5.4 序列视图编码

**`_pool_fixed(sequence, mask, output_tokens)`** —— 静态方法，定宽分箱的 mask 感知平均池化：

```python
source_length = sequence.get_shape().as_list()[1]      # 必须静态可知
width         = sequence.get_shape().as_list()[-1]
if source_length is None or width is None: raise ValueError("pooled sequence requires static shapes")
if source_length % output_tokens: raise ValueError("cannot pool {} tokens into {} bins")
bin_size = source_length // output_tokens
batch = tf.shape(sequence)[0]
grouped      = tf.reshape(sequence * tf.expand_dims(mask, -1), [batch, output_tokens, bin_size, width])
grouped_mask = tf.reshape(mask, [batch, output_tokens, bin_size])
count  = tf.reduce_sum(grouped_mask, axis=2, keepdims=True)          # [B, out, 1]
pooled = tf.reduce_sum(grouped, axis=2) / tf.maximum(count, 1.0)     # [B, out, W]
pooled_mask = tf.cast(tf.squeeze(count, axis=-1) > 0.0, tf.float32)  # [B, out]
return pooled, pooled_mask
```

⚠️ **分母是桶内有效位数，不是桶宽**。全 padding 桶 → 分子 0 / 分母 1 = **精确 0**，且 `pooled_mask=0`。这必不可少：padding 位经过 dense 后带有 bias + position + type embedding，是**非零**的。
💡 一次 reshape 取代逐桶 Python 循环（历史版本的 `_adaptive_pool` 循环 `output_tokens` 次，约 12 op/桶）。整除时两者**逐比特等价**。
⚠️ 返回的 mask 是 **float32**，不是 bool —— 下游 `fim_block.project_memory` 断言 mask 是 `[B,L]`，`masked_attention` 内部才 `tf.cast(valid, tf.float32)`。

**`_compress_sequence(group_name, start, end, output_tokens)`**：

```python
source = self.sequence_group_inputs[group_name][:, start:end, :]
raw_lengths = tf.cast(self.sequence_lengths[group_name], tf.int32)
if raw_lengths.get_shape().ndims == 2: raw_lengths = tf.squeeze(raw_lengths, -1)
source_length = end - start
valid_length  = tf.maximum(tf.minimum(raw_lengths - start, source_length), 0)   # ⚠️ offset 修正
source_mask   = self._mask(valid_length, source_length)
if source_length == output_tokens: return source, source_mask                   # 早返回，不池化
return self._pool_fixed(source, source_mask, output_tokens)
```

⚠️ **`valid_length` 必须减 `start`**：`long_click` 取 `[50:500]`，真实长度要先减 50 再 clamp 到 `[0,450]`，否则 mask 整体错位 50 位。

**`_sequence_stream(group_name, start, end, output_tokens, type_index)`**：

```python
sequence, mask = self._compress_sequence(group_name, start, end, output_tokens)
with tf.variable_scope("brv4_sequence_{}".format(group_name)):
    projected = tf.layers.dense(sequence, self.token_dim, name="projection")
    position = tf.get_variable("position_embedding", [1, output_tokens, self.token_dim],
                               initializer=tf.glorot_uniform_initializer())
    type_embedding = tf.get_variable("type_embedding", [1, 1, self.token_dim],
                                     initializer=tf.glorot_uniform_initializer())
    projected = projected + position + type_embedding * float(type_index + 1)
return projected, mask
```

⚠️ **variable_scope 按 `group_name` 划分，不是按 route 名**。5 条流各一套独立参数（scope 分别是 `brv4_sequence_user_click_seq` 等）。
⚠️ `position_embedding` 的第二维是 **`output_tokens`（池化后）**，不是原始序列长度。所以 `long_click` 读 450 步但 position 表只有 50 行。
⚠️ `type_embedding` 是 **per-stream 独立变量**（在 per-group scope 内创建，全仓无 `reuse=True`），`* float(type_index+1)` 的乘子**不省参数**，只是引入初始幅值不均衡：5 个 type_embedding 都用 `glorot_uniform` 初始化，乘子 1~5 使第 5 条流（upstream）的 type 偏移初始幅值是第 1 条流（click）的 5 倍。
💡 `tf.layers.dense` 用默认初始化（glorot_uniform kernel + zeros bias），与 `_project_token` 的显式 glorot 一致。

**`_build_sequence_tokens()`**：

```python
tokens = {}
for type_index, (name, group, start, end, stream_tokens, generic_tokens) in enumerate(SEQUENCE_SPECS):
    encoded = self._sequence_stream(group, start, end, stream_tokens, type_index)
    tokens[name] = encoded
    if generic_tokens != stream_tokens:
        tokens["{}_generic".format(name)] = self._pool_fixed(encoded[0], encoded[1], generic_tokens)
return tokens
```

产出的 6 个视图：

| key | 来源组 | 池化 | position 表 | type 乘子 | mask |
|---|---|---|---|---:|---|
| `click` | user_click_seq | 不池化（50→50） | `[1,50,128]` | 1 | `[B,50]` |
| `order` | user_order_seq | 不池化 | `[1,50,128]` | 2 | `[B,50]` |
| `long_click` | user_click_long_seq | **raw 450→50**（bin 9） | `[1,50,128]` | 3 | `[B,50]` |
| `atc` | user_cart_seq | 不池化（50→50） | `[1,50,128]` | 4 | `[B,50]` |
| `atc_generic` | ← 由 `atc` 派生 | **在已投影 token 上 50→25**（bin 2） | 无独立表（是 atc 的 pos[2i],pos[2i+1] 的平均） | 继承 4 | `[B,25]` |
| `upstream` | upstream_impression_seq | **raw 50→25**（bin 2） | `[1,25,128]` | 5 | `[B,25]` |

⚠️ **`atc_generic` 与 `upstream` 的池化位置不同**：`atc` 先在 native 50 上投影 + 加 position/type，**再**池化到 25；`upstream` 是**先**在 raw 上池化到 25，**再**投影 + 加自己的 position。这是刻意的 —— `atc` 必须保留 native 50 视图作为 dedicated route，而池化已编码的 token 可以复用同一份投影。重写时不要把两者统一。

💡 序列视图**只编码一次**，被所有 FIM 层复用（`build_fim_inputs` 里算，传给每层 `fim.call`）。每层自己拥有 K/V 投影参数。

**`_mean_sequence(group_name, size)`** 与 **`_masked_sequence_mean`**（供 user token 5/6 用）：

```python
sequence = self.sequence_group_inputs[group_name][:, :size, :]
lengths  = tf.minimum(self.sequence_lengths[group_name], size)
# _masked_sequence_mean:
mask  = tf.expand_dims(self._mask(lengths, max_len), -1)
total = tf.reduce_sum(sequence * mask, axis=1)
count = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
return tf.identity(total / count, name="brv4_{}_mean".format(group_name))
```

⚠️ 分母是**有效长度**，不是 `size`。且这里**不做归一化**（序列组不能用 `INormalization`：moments 会在 batch×time 上统计，把 padding 位算进去，平均有效长度 10/50 时 80% 统计样本是全 0 向量，会把 moving_mean 拉向 0、moving_var 压小）。

### 5.5 user token 构造（`_build_user_tokens`）

```python
user       = self._group("user")                                        # [B,272]
user_short = self._group("user_short")                                  # [B,904]
user_long  = self._group("user_long")                                   # [B,1656]
context    = tf.concat([self._group("context"), self._group("indicator_ctx_fea")],
                       axis=-1, name="brv4_context_features")           # [B,560]
click_mean = self._mean_sequence("user_click_seq", 50)                  # [B,176]
order_mean = self._mean_sequence("user_order_seq", 50)                  # [B,136]

seeds = [
    self._project_token(user,       "brv4_user_token",       input_norm=True),   # 272 ->128
    self._project_token(user_short, "brv4_user_short_token", input_norm=True),   # 904 ->128
    self._project_token(user_long,  "brv4_user_long_token",  input_norm=True),   # 1656->128
    self._project_token(context,    "brv4_context_token",    input_norm=True),   # 560 ->128
    self._project_token(self.scene_onehot_input, "brv4_scene_token", input_norm=True), # 10 ->128
    self._project_token(click_mean, "brv4_click_seq_seed",   input_norm=True),   # 176 ->128
    self._project_token(order_mean, "brv4_order_seq_seed",   input_norm=True),   # 136 ->128
]
user_global = self._project_token(
    tf.concat([user, user_short, user_long, context,
               self.scene_onehot_input, click_mean, order_mean],
              axis=-1, name="brv4_user_global_features"),               # [B,3714]
    "brv4_user_global_token", [self.tower_dim], input_norm=True)        # 3714->640
self._require_width(user_global, 640, "brv4_user_global_token")
user_global_chunks = tf.reshape(user_global,
    [tf.shape(user_global)[0], self.cache_chunks, self.token_dim],
    name="brv4_user_global_chunks")                                     # [B,5,128]
seeds.append(user_global_chunks[:, 0, :])
user_tokens = tf.stack(seeds, axis=1, name="brv4_user_tokens_8")        # [B,8,128]
assert user_tokens.get_shape().as_list()[1] == self.USER_TOKEN_COUNT
return user_tokens, user_global_chunks
```

**8 个 user token**：

| # | scope 名 | 输入 | 输入维 | 输出 |
|---|---|---|---:|---|
| 0 | `brv4_user_token` | `user` 组 | 272 | 128 |
| 1 | `brv4_user_short_token` | `user_short` 组 | 904 | 128 |
| 2 | `brv4_user_long_token` | `user_long` 组 | 1656 | 128 |
| 3 | `brv4_context_token` | `context` ⊕ `indicator_ctx_fea` | 560 | 128 |
| 4 | `brv4_scene_token` | `scene_onehot`（clip 后） | 10 | 128 |
| 5 | `brv4_click_seq_seed` | click 序列 masked mean | 176 | 128 |
| 6 | `brv4_order_seq_seed` | order 序列 masked mean | 136 | 128 |
| 7 | `brv4_user_global_token` 的 **chunk 0** | 上述 7 部分原始 concat | 3714 | 640→reshape→取 chunk 0 |

⚠️ **`user_global_chunks[:, 1:5, :]` 不是 user token**，它们流向 item 侧生成 4 个 fusion query（见 §5.7）。这是 user 信息注入 item 侧的通道。
⚠️ `user_global` 的输入是**原始 concat**（3714 维），不是 7 个 token 的拼接；且它自己带 `input_norm=True`，即对 3714 维整体做一次 `INormalization`，与各 token 分别归一化**不是同一套统计量**。
⚠️ `brv4_scene_token` 归一化了，但 `_scene_condition` 读的是**原始** `self.scene_onehot_input`，所以硬路由的 0/1 语义完好。

### 5.6 item cache 与训推分离（`_build_item_embedding`）

```python
cache_emb = self.assign_slot_embs                       # [B_i, 640]
self._require_width(cache_emb, 640, "brv4_item_cache")
if ego.is_training_mode():
    item_teacher = ego.DenseTower(
        name="brv4_item_cache_teacher", output_dims=[640],
        kernel_initializers=[tf.glorot_uniform_initializer()],
        bias_initializers=[tf.zeros_initializer()],
        activations=[None], norms=[True], use_bias=True,
    )(self._group("item"))                              # 816 -> 640
    item_emb = ego.replace_gradient(item_teacher, cache_emb, item_teacher)
else:
    item_emb = cache_emb                                # ⚠️ serving 完全旁路 816 维 item 特征
cache_norm = tf.sqrt(tf.reduce_sum(tf.square(item_emb), axis=-1, keepdims=True),
                     name="brv4_item_cache_norm")       # [B_i, 1]
return item_emb, cache_norm
```

💡 **训推分离机制**：训练态 `replace_gradient(teacher, cache_emb, teacher)` 让前向取 teacher 值、反向把梯度导向 teacher，同时 EGO 的 AssignOptimizer 把 teacher 输出**直接写入 slot 30102**；推理态只查表，60 个 item 特征（816 维）完全不参与计算图。
⚠️ **`item` 组不能加 `ego.Normalization` 之外的处理，也不能在 serving 侧引用**：`brv4_item_cache_teacher` 是训练态专属变量（522,880 参数），serving 图里不存在。
💡 `cache_norm` 是**监控量**，作为 `item_cache_norm` target 导出（见 §5.12），不参与打分。⚠️ 它是真实 norm，**不要**改成 `rsqrt` 形式。

### 5.7 SimTier 多模态特征（`utils/simtier_v1m4.py`）

**调用点**（`_build_multimodal_item_tokens`）：

```python
q_img   = self.group_inputs["target_item_query_img"]      # [B_i, 33]
q_title = self.group_inputs["target_item_query_title"]    # [B_i, 33]
short_sequences = [(self.sequence_group_inputs[n], self.sequence_lengths[n])
                   for n in IMAGE_SEQUENCE_GROUPS + TITLE_SEQUENCE_GROUPS]   # 6 路，顺序固定
long_sequences  = [(self.sequence_group_inputs[n], self.sequence_lengths[n])
                   for n in LONG_IMAGE_SEQUENCE_GROUPS]                     # 2 路
img_score, title_score = short_simtier(q_img, q_title, short_sequences, tiers=(5, 20))
long_image_score       = long_simtier(q_img, long_sequences, tiers=(20, 80))
self._require_width(img_score,  231, "brv4_item_image_score")
self._require_width(title_score, 231, "brv4_item_title_score")
self._require_width(long_image_score, 204, "brv4_item_long_image_simtier")
img_token   = self._project_token(img_score, "brv4_img_token")                        # 231->128
title_token = self._project_token(tf.concat([title_score, long_image_score], axis=-1),
                                  "brv4_title_token")                                 # 435->128
return img_token, title_token
```

⚠️ **`short_sequences` 的顺序必须是 `(clk_img, cart_img, order_img, clk_title, cart_title, order_title)`**，因为 `short_simtier` 内部用 `tf.gather(stack([q_img,q_title]), [0,0,0,1,1,1])` 配对 query，且输出按 `features[:, :3]` / `features[:, 3:]` 切分 image/title。

**`request_cosines(queries, keys, lengths=None, scope_name=...)`**：

```python
_, groups, length, width = keys.shape.as_list()          # 必须静态
with tf.name_scope(scope_name) as scope:
    if not tf.executing_eagerly():
        tf.compat.v1.add_to_collection(_ATTENTION_SCOPES, scope)   # ⚠️ 注册编译保护
    requests = tf.shape(keys)[0]
    query  = tf.reshape(queries, [-1, requests, groups, 1, width])
    memory = tf.identity(keys, name="memory")
    scores = tf.reduce_sum(query * memory[None], axis=-1)          # [C,R,G,L]
    valid = None
    if lengths is not None:
        lens = tf.clip_by_value(tf.cast(tf.reshape(lengths, [-1, groups]), tf.int32), 0, length)
        request_valid = tf.cast(tf.sequence_mask(lens, maxlen=length), tf.float32)
        valid = tf.reshape(request_valid[None] * tf.ones_like(scores), [-1, groups, length])
    scores = tf.reshape(scores, [-1, groups, length], name="cosine")
    return scores, valid
```

⚠️ **必须用 FP32 elementwise 乘 + `reduce_sum`，不能换 `matmul`**：硬分箱边界附近 ~1e-7 的差异会**改变直方图桶归属**，输出不是"近似相等"而是"个别样本某个桶计数差 1"。
⚠️ **`l2_normalize` 的 epsilon 必须是 `1e-6`**（`_L2_EPSILON`），不是参考实现的 `1e-3`。
💡 history 保持 `[R,G,L,D]` 不被 tile，靠 `memory[None]` 的广播对齐 `[C,R]`；scope 注册进 `_ATTENTION_SCOPES` 后由 `compile_with_request_attention` 保护。

**`_histogram_counts(indices, num_bins, valid_mask=None)`** —— 训练/serving 双实现：

```python
indices = tf.cast(indices, tf.int32)
valid   = tf.logical_and(indices >= 0, indices < num_bins)
weights = tf.cast(valid, tf.float32)
if valid_mask is not None: weights = weights * tf.cast(valid_mask, tf.float32)
if ego.is_training_mode():
    rows = tf.shape(indices)[0]
    safe_indices = tf.where(valid, indices, tf.zeros_like(indices))
    row_offset   = tf.reshape(tf.range(rows, dtype=tf.int32) * num_bins, [-1, 1])
    segment_ids  = tf.reshape(safe_indices + row_offset, [-1])
    counts = tf.math.unsorted_segment_sum(tf.reshape(weights, [-1]), segment_ids, rows * num_bins)
    counts = tf.reshape(counts, [-1, num_bins])
else:
    chunks = []
    for start in range(0, num_bins, 8):
        bins = tf.constant(list(range(start, min(start + 8, num_bins))), tf.int32)
        matches = tf.cast(tf.equal(indices[:, None, :], bins[None, :, None]), tf.float32)
        chunks.append(tf.reduce_sum(matches * weights[:, None, :], axis=2))
    counts = tf.concat(chunks, axis=1)
return counts
```

💡 **训练走 `unsorted_segment_sum`（线性时间，不物化 `[rows,bins,L]`），serving 走 8-bin 分块 Equal**。原因：tf2onnx 把 `UnsortedSegmentSum` 经 `Unique` 下降，TRT 无法解析；而 `ScatterNd` 的下降缺少重复 bin 所需的 additive reduction，也不是替代方案。训练图不导出 ONNX，所以 Unique 无害。
⚠️ 三种实现（Equal / segment_sum / 分块 Equal）产生**完全相同的整数计数**，因为被累加值恒为精确的 0.0/1.0 且计数 ≤ 500，fp32 下是精确整数。

**`_coarse_from_fine(counts, coarse, fine)`** —— 层级派生：

```python
if fine % coarse: raise ValueError("fine tier must be a multiple of coarse tier")
groups = counts.shape.as_list()[1]                       # 必须静态
coarse_counts = tf.reduce_sum(tf.reshape(counts[..., :fine],
                              [-1, groups, coarse, fine // coarse]), axis=-1)
return tf.concat([coarse_counts, counts[..., fine:]], axis=-1)   # [.,G,coarse+1]
```

⚠️ **精确等价的前提**：`fine % coarse == 0`（20%5=0、80%20=0），且 `fl(x·fine) == (fine/coarse)·fl(x·coarse)` 精确成立（因为 `fine/coarse = 4 = 2²` 是 2 的幂，乘以 2 的幂与正确舍入可交换）。由此 `floor(floor(a)/4) == floor(b)`。`cos==1` 的上界桶由 `concat(counts[..., fine:])` 单独保留。

**`short_simtier(query_image, query_title, short_sequences, tiers=(5,20))`**：

```python
assert len(short_sequences) == 6
coarse, fine = tiers                                     # 5, 20
keys    = tf.stack([v for v, _ in short_sequences], axis=1)              # [R,6,50,33]
queries = tf.gather(tf.stack([query_image, query_title], axis=1), [0,0,0,1,1,1], axis=1)
keys    = tf.nn.l2_normalize(keys,    axis=-1, epsilon=1e-6)
queries = tf.nn.l2_normalize(queries, axis=-1, epsilon=1e-6)
length  = keys.shape.as_list()[2]                        # 50，必须静态
cosines, _ = request_cosines(queries, keys, scope_name="brv4_short_cosines")   # ⚠️ lengths=None
indices = tf.cast((cosines + 1.0) / 2.0 * float(fine), tf.int32)               # ⚠️ 不 clip cosines
counts  = _histogram_counts(tf.reshape(indices, [-1, length]), fine + 1)        # ⚠️ valid_mask=None
counts  = tf.reshape(counts, [-1, 6, fine + 1])
histogram = tf.concat([_coarse_from_fine(counts, coarse, fine), counts], axis=-1)
histogram /= float(length)                               # ⚠️ 固定除以 50，不是有效长度
route_width = length + (coarse + 1) + (fine + 1)         # 50 + 6 + 21 = 77
features = tf.concat([cosines, histogram], axis=-1)      # [B,6,77]
image_score = tf.reshape(features[:, :3], [-1, 3 * route_width])    # [B,231]
title_score = tf.reshape(features[:, 3:], [-1, 3 * route_width])    # [B,231]
return image_score, title_score
```

⚠️ **短路径的三条语义必须原样保留**（它们与长路径不同，是历史 `din_attention_cos(use_agg=False)` 的行为）：
1. **不做长度 mask**（`lengths=None`、`valid_mask=None`）—— 全部 50 个位置都计入，包括 padding（padding 的 cosine=0 → 落在桶 `fine/2`）
2. **分母是固定的 50**，不是有效长度
3. **cosines 不 clip**（长路径 clip）

输出布局是 **route-major `[cos(50), coarse(6), fine(21)]` × 3 route = 231**，与历史 `get_item_attention_embs` 的 `[cos, tier]` 逐路拼接完全一致。

**`long_simtier(query_image, long_sequences, tiers=(20,80))`**：

```python
assert len(long_sequences) == 2
coarse, fine = tiers                                     # 20, 80
keys    = tf.stack([v for v, _ in long_sequences], axis=1)               # [R,2,500,33]
lengths = tf.stack([l for _, l in long_sequences], axis=1)               # [R,2]
keys    = tf.nn.l2_normalize(keys, axis=-1, epsilon=1e-6)
queries = tf.nn.l2_normalize(query_image, axis=-1, epsilon=1e-6)
queries = tf.tile(queries[:, None, :], [1, 2, 1])                        # [B,2,33]
length  = keys.shape.as_list()[2]                        # 500
cosines, valid = request_cosines(queries, keys, lengths=lengths,
                                 scope_name="brv4_long_cosines")
cosines = tf.clip_by_value(cosines, -1.0, 1.0)                           # ⚠️ 长路径 clip
indices = tf.cast((cosines + 1.0) * (float(fine) / 2.0), tf.int32)       # (cos+1)*40
indices = tf.clip_by_value(indices, 0, fine)                             # ⚠️ clip 到 [0,80]
counts  = _histogram_counts(tf.reshape(indices, [-1, length]), fine + 1,
                            valid_mask=tf.reshape(valid, [-1, length]))
counts  = tf.reshape(counts, [-1, 2, fine + 1])
histogram = tf.concat([_coarse_from_fine(counts, coarse, fine), counts], axis=-1)  # [B,2,102]
normalizer = tf.maximum(tf.reduce_sum(valid, axis=-1, keepdims=True), 1.0)         # [B,2,1]
histogram /= normalizer                                # ⚠️ 按有效长度归一化
return tf.reshape(histogram, [-1, 2 * ((coarse + 1) + (fine + 1))])      # [B,204]
```

⚠️ 长路径的索引公式是 `(cos+1) * (fine/2)`，短路径是 `(cos+1)/2 * fine` —— **数值相同但书写不同**，两者都保持历史实现的写法以确保 fp 逐位一致。
输出是 **group-major `[coarse(21), fine(81)]` × 2 group = 204**，与历史 `tf.concat([per-group 102], axis=-1)` 一致。

### 5.8 item token 构造与 fusion query

**`_build_item_tokens(item_emb, user_global_chunks)`**：

```python
item_chunks = tf.reshape(item_emb, [tf.shape(item_emb)[0], self.cache_chunks, self.token_dim],
                         name="brv4_item_cache_chunks")                 # [B_i,5,128]
item_token   = item_chunks[:, 0, :]                                     # chunk 0
recall_token = self._project_token(self._group("recall"), "brv4_recall_token", input_norm=True)
img_token, title_token = self._build_multimodal_item_tokens()
fused_queries = self._build_fusion_queries(user_global_chunks, item_chunks)
item_tokens = tf.stack([item_token, recall_token, img_token, title_token]
                       + [fused_queries[r] for r in FUSION_ROUTES],
                       axis=1, name="brv4_item_tokens_8")                # [B_i,8,128]
assert item_tokens.get_shape().as_list()[1] == self.ITEM_TOKEN_COUNT
```

**8 个 item token**：

| # | 名称 | 来源 | 输入维 | batch 粒度 |
|---|---|---|---:|---|
| 0 | `item_token` | item cache 640 的 **chunk 0** | — | 候选 |
| 1 | `brv4_recall_token` | `recall` 组（input_norm） | 80 | 候选 |
| 2 | `brv4_img_token` | SimTier short image | 231 | 候选 |
| 3 | `brv4_title_token` | SimTier short title ⊕ long image | 231+204=435 | 候选 |
| 4 | fusion query `click` | `user_global_chunks[:,1]` ⊕ `item_chunks[:,1]` | 128+128 | 候选（user 部分请求级） |
| 5 | fusion query `atc` | `[:,2]` ⊕ `[:,2]` | 256 | 同上 |
| 6 | fusion query `order` | `[:,3]` ⊕ `[:,3]` | 256 | 同上 |
| 7 | fusion query `long_click` | `[:,4]` ⊕ `[:,4]` | 256 | 同上 |

⚠️ **token 4..7 的顺序必须严格等于 `FUSION_ROUTES = ("click","atc","order","long_click")`**，因为 FIM 的 `_item_ca` 把 `query[:, 4:]` 按这个顺序对到 `dedicated_k`（由 `tf.stack([key_routes[r] for r in FUSION_ROUTES], axis=1)` 构造）。顺序错位不会报错，只会静默地把 query 接到错误的行为序列上。
⚠️ **token 0..3 是 base token**，`_item_ca` 用 `query[:, :ITEM_BASE_TOKEN_COUNT]`（即 `[:4]`）对 generic memory。

**`_build_fusion_queries(user_global_chunks, item_chunks)`**：

```python
queries = {}
with tf.variable_scope("brv4_fusion_queries"):
    for index, route in enumerate(FUSION_ROUTES, start=1):        # index = 1,2,3,4
        with tf.variable_scope("{}_query".format(route)):
            kernel = tf.get_variable("kernel", [2 * self.token_dim, self.token_dim],
                                     initializer=tf.glorot_uniform_initializer())   # [256,128]
            bias   = tf.get_variable("bias", [self.token_dim],
                                     initializer=tf.zeros_initializer())
        user_projection = tf.matmul(user_global_chunks[:, index, :], kernel[: self.token_dim, :],
                                    name="brv4_{}_user_projection".format(route))
        item_projection = tf.matmul(item_chunks[:, index, :], kernel[self.token_dim :, :],
                                    name="brv4_{}_item_projection".format(route))
        queries[route] = tf.identity(item_projection + user_projection + bias,
                                     name="{}_query".format(route))
return queries
```

⚠️ **保留完整的 `[256,128]` 变量，只在计算时切片**。这样 (a) glorot 的 fan-in 仍是 256，初始方差不变；(b) 变量名 `brv4_fusion_queries/{route}_query/{kernel,bias}` 与历史 `tf.layers.dense(pair, 128, name="{route}_query")` 完全一致，checkpoint 直接可映射。
💡 **`item_projection + user_projection + bias` 这个加法就是粒度对齐点**：user 投影在请求 batch 上算完，EGO 在加法处对齐 COMMON 行。数学上是 `concat([u,i]) @ W + b = u @ W[:128] + i @ W[128:] + b` 的恒等式。
⚠️ 加法顺序是 `item + user + bias`（左结合），不是 `user + item + bias`。数学等价，fp 舍入不同。

### 5.9 FIM block（`models/fim_block.py`）

#### 5.9.1 模块级 helper

```python
def grouped_linear(inputs, kernel, name=None):
    """[B, *groups, D] @ [*groups, D, H] -> [B, *groups, H]."""
    rank = inputs.shape.ndims                      # 必须 >= 3 且与 kernel 同 rank
    with tf.name_scope(name or "grouped_linear"):
        grouped   = tf.transpose(inputs, list(range(1, rank - 1)) + [0, rank - 1])
        projected = tf.matmul(grouped, kernel)     # 矩阵行维是 B，权重不在 B 上广播
        output    = tf.transpose(projected, [rank - 2] + list(range(rank - 2)) + [rank - 1])
        return tf.identity(output, name="output")

def _silu(x): return x * tf.nn.sigmoid(x)

def grouped_glorot(shape, dtype=None, fan_in=None, **kwargs):
    if len(shape) < 3: raise ValueError("grouped_glorot requires grouped matrix weights")
    fan_in = int(shape[-2]) if fan_in is None else fan_in
    limit  = math.sqrt(6.0 / (fan_in + int(shape[-1])))
    return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype or tf.float32)

def rms_norm(x, scale, eps=1e-6):
    inv_rms = tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + eps)
    return x * inv_rms * scale                     # ⚠️ rsqrt + 乘法，不是 sqrt + 除法
```

⚠️ `grouped_glorot` 的 `fan_in` 可被覆盖：当一个逻辑上 `[2D, H]` 的 kernel 被拆成两个 `[D, H]` 时，必须传 `fan_in=2*D` 才能保持初始方差（本模型未用到，但重写时若拆分要注意）。

#### 5.9.2 `GroupedSwiGLU`

```python
class GroupedSwiGLU(tf.keras.layers.Layer):
    def __init__(self, num_tokens, d_model, hidden_dim, name):
        with tf.compat.v1.variable_scope(name):
            self.gate_up      = add_weight("gate_up",      [num_tokens, d_model, 2*hidden_dim], grouped_glorot)
            self.gate_up_bias = add_weight("gate_up_bias", [num_tokens, 2*hidden_dim],          zeros)
            self.down         = add_weight("down",         [num_tokens, hidden_dim, d_model],  grouped_glorot)
            self.down_bias    = add_weight("down_bias",    [num_tokens, d_model],              zeros)

    def call(self, x, token_offset=0):
        count = x.shape.as_list()[1]               # 必须静态
        end   = token_offset + count
        gate_up = grouped_linear(x, self.gate_up[token_offset:end]) + self.gate_up_bias[token_offset:end]
        gate, up = tf.split(gate_up, 2, axis=-1)
        output = grouped_linear(_silu(gate) * up, self.down[token_offset:end]) + self.down_bias[token_offset:end]
        return output                              # ⚠️ 不含残差，残差在调用处加
```

💡 **user 与 item 共用同一个 `GroupedSwiGLU` 实例**（16 个 token 的参数），靠 `token_offset` 切片：user 用 `[0:8]`，item 用 `[8:16]`。`s_ffn` 与 `m_ffn` 是两个独立实例。
💡 gate 与 up 融合成一个 `[D, 2H]` matmul 再 `split`，比两个独立 matmul 少一次 launch。

#### 5.9.3 `UIHeadMixer`（无参数的通道置换）

```python
class UIHeadMixer(object):
    def __init__(self, num_tokens, d_model, user_tokens):     # (16, 128, 8)
        if num_tokens != 2 * user_tokens: raise ValueError(...)
        if d_model % num_tokens: raise ValueError(...)
        self.half_dim = d_model // 2        # 64
        depth = d_model // num_tokens       # 8
        user_indices = [source*d_model + half*self.half_dim + target*depth + channel
                        for target in range(user_tokens) for half in range(2)
                        for source in range(user_tokens) for channel in range(depth)]      # 1024 个
        item_indices = [source*d_model + self.half_dim + target*depth + channel
                        for target in range(user_tokens) for source in range(user_tokens)
                        for channel in range(depth)]                                       # 512 个
        self.user_indices      = tf.constant(user_indices, tf.int32, name="ui_user_permutation")
        self.item_tail_indices = tf.constant(item_indices, tf.int32, name="ui_item_tail_permutation")

    def mix_user(self, user_input):          # [B,8,128] -> [B,8,128]
        flat = tf.reshape(user_input, [tf.shape(user_input)[0], self.user_tokens * self.d_model])
        return tf.reshape(tf.gather(flat, self.user_indices, axis=1), [-1, self.user_tokens, self.d_model])

    def mix_item_tail(self, item_input):     # [B,8,128] -> [B,8,64]
        flat = tf.reshape(item_input, [tf.shape(item_input)[0], self.user_tokens * self.d_model])
        return tf.reshape(tf.gather(flat, self.item_tail_indices, axis=1), [-1, self.user_tokens, self.half_dim])
```

💡 预计算常量索引 + **单次 `tf.gather`** 完成双半区置换，取代 split/reshape/concat/zero-mask 的多算子写法。
⚠️ `mix_user` 的输出是 `[B,8,128]`（两个半区都混合），`mix_item_tail` 只输出**后半区** `[B,8,64]`（item 的前半区由 `user_tail` 填充，见 §5.9.5）。
⚠️ 索引列表的嵌套循环顺序（`target → half → source → channel` 与 `target → source → channel`）决定了置换语义，**不能改动顺序**。

#### 5.9.4 `FIMBlock.__init__` 的参数

`FIMBlock(layer_idx, d_model=128, num_heads=4, hidden_dim=128, name=None)`，`name` 默认 `"brv4_fim_{layer_idx}"`，是 `tf.keras.layers.Layer`。`self.depth = 32`，`self.scale = 32 ** -0.5 = 0.17677669529663687`。

全部在 `with tf.compat.v1.variable_scope(self.name)` 内用 `add_weight` 创建：

| 变量 | shape | 初始化 |
|---|---|---|
| `route_kv_norm_scale` | `[6, 128]` | ones |
| `route_kv_kernel` | `[6, 128, 256]` | `grouped_glorot` |
| `route_kv_bias` | `[6, 256]` | zeros |
| `q_norm_scale` | `[16, 128]` | ones |
| `q_kernel` | `[16, 128, 128]` | `grouped_glorot` |
| `o_kernel` | `[16, 128, 128]` | `grouped_glorot` |
| `o_bias` | `[16, 128]` | zeros |
| `s_norm_scale` | `[16, 128]` | ones |
| `mix_norm_scale` | `[16, 128]` | ones |
| `ffn_norm_scale` | `[16, 128]` | ones |
| `s_ffn/{gate_up,gate_up_bias,down,down_bias}` | `[16,128,256]` `[16,256]` `[16,128,128]` `[16,128]` | grouped_glorot / zeros |
| `m_ffn/...` | 同上 | 同上 |

单层可训练参数 **2,318,592**，2 层共 **4,637,184**（占全模型 40.5%）。

#### 5.9.5 `project_memory` 与 attention

```python
def project_memory(self, sequence_tokens):
    key_routes, value_routes, masks = {}, {}, {}
    for route_idx, route in enumerate(self.route_names):        # SEQUENCE_ROUTES，6 路
        sequence, mask = sequence_tokens[route]
        # 断言 sequence 是 [B, ROUTE_LENGTHS[route], 128]，mask 是 [B, ROUTE_LENGTHS[route]]
        with tf.name_scope("brv4_fim_{}_{}_kv".format(self.layer_idx, route)):
            normalized = rms_norm(sequence, self.route_kv_norm[route_idx])
            kv = tf.matmul(normalized, self.route_kv_kernel[route_idx]) + self.route_kv_bias[route_idx]
        key, value = tf.split(kv, 2, axis=-1)
        key_routes[route]   = self._split_heads(key)             # [B,H,L,32]
        value_routes[route] = self._split_heads(value)
        masks[route] = mask
    return {
        "dedicated_k":    tf.stack([key_routes[r]   for r in FUSION_ROUTES],   axis=1),   # [B,4,H,50,32]
        "dedicated_v":    tf.stack([value_routes[r] for r in FUSION_ROUTES],   axis=1),
        "dedicated_mask": tf.stack([masks[r]        for r in FUSION_ROUTES],   axis=1),   # [B,4,50]
        "generic_k":      tf.concat([key_routes[r]   for r in GENERIC_ROUTES], axis=2),   # [B,H,200,32]
        "generic_v":      tf.concat([value_routes[r] for r in GENERIC_ROUTES], axis=2),
        "generic_mask":   tf.concat([masks[r]        for r in GENERIC_ROUTES], axis=1),   # [B,200]
    }
```

💡 **同一份 per-route 张量同时派生 dedicated(`stack`) 与 generic(`concat`) 两种布局**，K/V 只投影一次。每层拥有自己的 `route_kv_*` 参数，但 6 个编码视图（§5.4 的产物）跨层共享。
⚠️ generic memory 的拼接顺序是 `GENERIC_ROUTES = (click, order, long_click, atc_generic, upstream)`，长度 50+50+50+25+25 = **200**。
⚠️ `atc`（native 50）只进 dedicated，不进 generic；进 generic 的是 `atc_generic`（25）。

**`masked_attention(query, key, value, valid, scale)`**（`utils/request_attention.py`）：

```python
logits  = tf.matmul(query, key, transpose_b=True, name="query_key") * scale
weights = tf.nn.softmax(logits + (1.0 - valid) * -1.0e9, axis=-1)
weights = weights * valid                                                    # 二次屏蔽
weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1.0e-9) # 二次归一化
return tf.matmul(weights, value, name="attention_value"), weights
```

⚠️ **三重防御缺一不可**：`-1e9` 屏蔽在整条序列全空时会失效（所有 logits 变成同一常数，softmax 退化成均匀分布而非 0），所以必须再 `* valid`；乘完变成 `0/0`，而 IEEE 754 下 `0.0/0.0 = NaN`，所以必须 `+ 1.0e-9`。空序列（`long_click` 的 `[50:500]` 对长期点击 ≤50 的用户、无购物车/无订单/无上游曝光）是**线上高频路径**。
💡 `weights / (sum + 1e-9)` 与 `weights / max(sum, 1e-9)` 在 fp32 下**逐比特相同**（`1.0 + 1e-9` 舍回 `1.0`），但前者少一个算子。

**`request_shared_attention(query, key, value, mask, num_heads, dedicated=False, collect_monitor=True)`**：serving 专用，把 memory 保持在请求粒度。注册 scope `uniformer_request_attention` 进 `_ATTENTION_SCOPES`；`requests = tf.shape(key)[0]`；`q = tf.reshape(query, [-1, requests, tokens, num_heads, depth])`；
- `dedicated=True`：`q = transpose(q, [1,2,3,0,4])` → `[R,T,H,C,d]`，`valid = mask[:,:,None,None,:]`
- `dedicated=False`：`q = transpose(q, [1,3,0,2,4])` 再 `reshape [R,H,-1,d]`，`valid = mask[:,None,None,:]`

然后 `k = tf.identity(key, name="memory_key")`、`v = tf.identity(value, name="memory_value")`，调 `masked_attention(q,k,v,valid, depth**-0.5)`，可选算 entropy，最后逆变换回 `[-1, tokens, width]`。

⚠️ **前提：EGO GraphEditor 用 `tf.tile([C,1,...])` 广播 COMMON 行，所以 `B = C*R` 的行序是 `[candidate_group, request]` 而非 `[request, candidate_group]`**；且**所有请求的候选数 C 必须相同**（`reshape` 依赖整除，不满足时报错而非静默错配）。

**两个 attention 包装**：

```python
def _generic_attention(self, query, key, value, mask, broadcast_memory=False, collect_monitor=False):
    if broadcast_memory:
        return request_shared_attention(query, key, value, mask, self.num_heads,
                                        dedicated=False, collect_monitor=collect_monitor)
    batch, token_count = tf.shape(query)[0], query.shape.as_list()[1]
    query = tf.transpose(tf.reshape(query, [batch, token_count, self.num_heads, self.depth]), [0,2,1,3])
    valid = tf.cast(mask[:, None, None, :], tf.float32)
    context, weights = masked_attention(query, key, value, valid, self.scale)
    context = tf.reshape(tf.transpose(context, [0,2,1,3]), [batch, token_count, self.d_model])
    entropy = None
    if collect_monitor:
        entropy = -tf.reduce_mean(tf.reduce_sum(weights * tf.math.log(weights + 1.0e-9), axis=-1))
    return context, entropy

def _dedicated_attention(self, query, key, value, mask, broadcast_memory=False, collect_monitor=False):
    if broadcast_memory:
        return request_shared_attention(query, key, value, mask, self.num_heads,
                                        dedicated=True, collect_monitor=collect_monitor)
    batch = tf.shape(query)[0]
    query = tf.reshape(query, [batch, 4, self.num_heads, 1, self.depth])    # ⚠️ 4 路进 batch 维
    valid = tf.cast(mask[:, :, None, None, :], tf.float32)                  # [B,4,1,1,50]
    context, weights = masked_attention(query, key, value, valid, self.scale)
    context = tf.reshape(context, [batch, 4, self.d_model])
    ...
```

💡 **4 路 dedicated 合成一次 batched attention**（`[B,4,H,1,d] × [B,4,H,50,d]`），而不是 Python 循环 4 次。前提是 4 路长度相同 —— `FUSION_ROUTES` 的 click/atc/order/long_click **全是 50**。
⚠️ `_dedicated_attention` 里的 `4` 是硬编码的，等于 `len(FUSION_ROUTES)`；改路由数必须同步。

#### 5.9.6 `call` 的完整顺序

```python
def call(self, user_tokens, item_tokens, sequence_tokens, training=True, collect_monitor=False):
    self._check_tokens(user_tokens, item_tokens)          # 两侧都必须静态是 [B,8,128]
    memory = self.project_memory(sequence_tokens)

    # ---- user 分支（全程请求粒度）----
    user_hs, user_context, user_entropy, user_s_delta = self._user_ca(user_tokens, memory, collect_monitor)
    user_mixed = user_hs + self.mixer.mix_user(rms_norm(user_hs, self.mix_norm[:8]))
    half = self.d_model // 2                              # 64
    user_tail = tf.identity(user_mixed[:, :, half:], name="fim_{}_user_mixed_tail".format(self.layer_idx))
    user_m_delta = self.m_ffn.call(rms_norm(user_mixed, self.ffn_norm[:8]), token_offset=0)
    user_final = user_mixed + user_m_delta

    # ---- item 分支（候选粒度）----
    (item_hs, item_context, dedicated_entropy,
     item_generic_entropy, item_s_delta) = self._item_ca(
         item_tokens, memory, broadcast_memory=not training, collect_monitor=collect_monitor)
    item_tail = self.mixer.mix_item_tail(rms_norm(item_hs, self.mix_norm[8:]))
    item_mixed = tf.concat([item_hs[:, :, :half] + user_tail,
                            item_hs[:, :, half:] + item_tail],
                           axis=-1, name="fim_{}_item_mixed".format(self.layer_idx))
    item_m_delta = self.m_ffn.call(rms_norm(item_mixed, self.ffn_norm[8:]), token_offset=8)
    item_final = item_mixed + item_m_delta

    monitor = {...} if collect_monitor else {}
    return user_final, item_final, monitor
```

其中：

```python
def _user_ca(self, user_tokens, kv_cache, collect_monitor=False):
    query = self._project_query(user_tokens, 0)                    # 用 q_norm/q_kernel 的 [0:8]
    context, entropy = self._generic_attention(                    # ⚠️ 不传 broadcast_memory
        query, kv_cache["generic_k"], kv_cache["generic_v"],
        kv_cache["generic_mask"], collect_monitor=collect_monitor)
    hidden, s_delta = self._post_ca(user_tokens, context, 0)
    return hidden, context, entropy, s_delta

def _item_ca(self, item_tokens, kv_cache, broadcast_memory, collect_monitor=False):
    query = self._project_query(item_tokens, self.user_tokens)     # 用 [8:16]
    dedicated, ded_ent = self._dedicated_attention(
        query[:, ITEM_BASE_TOKEN_COUNT:, :], kv_cache["dedicated_k"],
        kv_cache["dedicated_v"], kv_cache["dedicated_mask"],
        broadcast_memory=broadcast_memory, collect_monitor=collect_monitor)
    generic, gen_ent = self._generic_attention(
        query[:, :ITEM_BASE_TOKEN_COUNT, :], kv_cache["generic_k"],
        kv_cache["generic_v"], kv_cache["generic_mask"],
        broadcast_memory=broadcast_memory, collect_monitor=collect_monitor)
    context = tf.concat([generic, dedicated], axis=1)              # ⚠️ generic 在前，与 token 0..3 / 4..7 对应
    hidden, s_delta = self._post_ca(item_tokens, context, self.user_tokens)
    return hidden, context, ded_ent, gen_ent, s_delta

def _post_ca(self, tokens, context, token_offset):
    end = token_offset + tokens.shape.as_list()[1]
    projected = grouped_linear(context, self.o_kernel[token_offset:end]) + self.o_bias[token_offset:end]
    h_o = tokens + projected                                       # 残差 1
    s_delta = self.s_ffn.call(rms_norm(h_o, self.s_norm[token_offset:end]), token_offset=token_offset)
    return h_o + s_delta, s_delta                                  # 残差 2

def _project_query(self, tokens, token_offset):
    count = tokens.shape.as_list()[1]
    end = token_offset + count
    normalized = rms_norm(tokens, self.q_norm[token_offset:end])
    return grouped_linear(normalized, self.q_kernel[token_offset:end])
```

⚠️ **每层内的算子顺序是 PreLN 结构**：`CA → +残差 → S-FFN(rms_norm 前置) → +残差 → UI mix → +残差 → M-FFN(rms_norm 前置) → +残差`。共 4 次残差加法。
⚠️ **`user_tail` 必须在 user M-FFN 之前取出**（注释原文："Transfer the residual-inclusive tail, before the user M-FFN"），且它带残差（是 `user_mixed` 的后半，不是 `user_hs` 的后半）。这样 EGO 只需在这一个加法点对齐 COMMON 行。
⚠️ **item 的前半通道加 `user_tail`、后半通道加 `item_tail`** —— 这是 U/I 信息交换的唯一通道，且是**单向**的：user 分支的 `mix_user` 索引只在 `range(user_tokens)` 内取，`_user_ca` 只 attend 序列 memory，所以 **user token 从头到尾看不到任何 item 信息**。这正是 user 侧能保持请求粒度的根本原因。
⚠️ `_user_ca` **不传** `broadcast_memory`（默认 False）—— user 侧本来就是请求粒度，无需 request-shared 路径。只有 `_item_ca` 传 `not training`。
💡 `collect_monitor` 默认 False，此时 `monitor` 是空 dict，6 项指标（dedicated/generic attention entropy、item/user context norm、s_ffn/m_ffn delta norm）都不计算。`build_esmm` 当前**没有传** `collect_monitor=True`，所以监控是关闭的。

### 5.10 `rank_global`（`_build_rank_global`）

```python
user_flat = tf.reshape(user_tokens, [-1, 8 * 128], name="brv4_rank_user_tokens_flat")   # [B_u,1024]
item_flat = tf.reshape(item_tokens, [-1, 8 * 128], name="brv4_rank_item_tokens_flat")   # [B_i,1024]
user_vector = self._project_token(user_flat, "brv4_rank_user_a", input_norm=True)       # 1024->128
item_vector = self._project_token(item_flat, "brv4_rank_item_b", input_norm=True)       # 1024->128

with tf.variable_scope("brv4_rank_global_mlp"):
    with tf.variable_scope("layer_1"):
        kernel_1 = tf.get_variable("kernel", [256, 256], initializer=tf.glorot_uniform_initializer())
        bias_1   = tf.get_variable("bias",   [256],      initializer=tf.zeros_initializer())
    with tf.variable_scope("layer_2"):
        kernel_2 = tf.get_variable("kernel", [256, 256], initializer=tf.glorot_uniform_initializer())
        bias_2   = tf.get_variable("bias",   [256],      initializer=tf.zeros_initializer())

user_projection = tf.matmul(user_vector, kernel_1[: self.token_dim, :], name="brv4_rank_user_projection")
item_projection = tf.matmul(item_vector, kernel_1[self.token_dim :, :], name="brv4_rank_item_projection")
first = item_projection + user_projection + bias_1
hidden = tf.nn.leaky_relu(first)                        # ⚠️ 激活必须在两部分相加与 bias 之后
rank_global = tf.matmul(hidden, kernel_2) + bias_2
return tf.identity(rank_global, name="brv4_rank_global")   # [B_i, 256]
```

⚠️ 变量名刻意做成 `brv4_rank_global_mlp/layer_{1,2}/{kernel,bias}`，与历史 `DenseTower(name="brv4_rank_global_mlp", output_dims=[256,256], activations=[leaky_relu, None], norms=[False,False])` 产生的名字一致（EGO `DenseTower` 用 `K.Dense(name="{tower}/layer_{i}")`），保证 checkpoint 可映射。
⚠️ 第二层 `[256,256]` **完全不动**，只拆第一层。
💡 这是 `rank_global` 而非历史的 `sum/max` 聚合：把 8 个 token 展平成 1024 维后各自投影到 128，再拼接过两层 MLP。user 侧与 item 侧**分开投影**（`brv4_rank_user_a` / `brv4_rank_item_b`），所以 user 那一路在请求粒度算。

### 5.11 TIM block（`models/tim_block.py`）

#### 5.11.1 入口与常量

```python
EXPERT_COUNT = 2

def _build_tim(self, user_tokens, item_tokens, training):
    return ParallelTIMBlock(
        d_model=self.token_dim,          # 128
        num_heads=self.tim_heads,        # 8   -> depth = 16, scale = 16**-0.5 = 0.25
        hidden_dim=self.tim_hidden_dim,  # 128
        num_queries=len(self.TIM_TOKEN_NAMES),   # 7
        user_tokens=self.USER_TOKEN_COUNT,       # 8
        item_tokens=self.ITEM_TOKEN_COUNT,       # 8
    )(user_tokens, item_tokens, training=training)

# __call__:
def __call__(self, user_tokens, item_tokens, training=True):
    self._check_tokens(user_tokens, item_tokens)     # 两侧都必须静态是 [B,8,128]
    with tf.variable_scope("brv4_tim_parallel"):
        return self._forward(user_tokens, item_tokens, training)
```

`self.memory_tokens = 8 + 8 = 16`，`self.depth = 16`，`self.scale = 0.25`。

#### 5.11.2 模块级 helper

```python
def _shared_linear(x, kernel, bias, name=None):
    """一个 rank-2 kernel 作用于最后一维，展平后走 XLA 的 Dot 路径。"""
    with tf.name_scope(name or "shared_linear"):
        width  = kernel.get_shape().as_list()[-1]
        flat   = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])
        output = tf.matmul(flat, kernel) + bias
        return tf.reshape(output, tf.concat([tf.shape(x)[:-1], [width]], axis=0))

def _expert_linear(inputs, kernel):
    """[B,E,T,D] @ [E,D,H] -> [B,E,T,H]，一个 batched MatMul。"""
    _, experts, tokens, width = inputs.shape.as_list()      # 必须全静态
    output_width = kernel.shape.as_list()[-1]
    grouped   = tf.reshape(tf.transpose(inputs, [1,0,2,3]), [experts, -1, width])
    projected = tf.matmul(grouped, kernel)
    projected = tf.reshape(projected, [experts, -1, tokens, output_width])
    return tf.transpose(projected, [1,0,2,3])
```

从 `models/rankmixer_v2_layers.py` 导入的四个 helper：

```python
def _independent_kernels(base_name, count, input_dim, output_dim):
    """创建 count 个独立的 rank-2 变量 {base_name}_{i}，初始化 glorot_uniform。
    ⚠️ 绝不能用一个 rank-3 可训练变量：EGO XLA CPU sandbox 在 TransformMmt4D 上会崩溃
    （无论是被 batched matmul 消费，还是被切片成 rank-2）。"""
    return [tf.get_variable("{}_{}".format(base_name, i), [input_dim, output_dim],
                            initializer=tf.glorot_uniform_initializer())
            for i in range(count)]

def _token_independent_matmul(x, kernels):
    """[B,N,K] x N个[K,O] -> [B,N,O]。训练：N 个独立 rank-2 MatMul + Concat；
    serving：tf.stack(kernels) 后一个 batched matmul。"""

def apply_rms_norm(x, scale, epsilon=1e-6):
    inv_rms = tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)
    if scale.get_shape().ndims == 1:
        scale = tf.reshape(scale, [1, 1, scale.get_shape().as_list()[0]])
    else:
        scale = tf.expand_dims(scale, axis=0)      # [N,D] -> [1,N,D]
    return x * inv_rms * scale

def packed_per_token_swiglu(x, num_tokens, d_model, hidden_dim, name, residual=True):
    """N 个独立 SwiGLU，两个 BatchMatMul。内部已含 rms_norm 与残差，调用处不重复加。"""
    with tf.variable_scope(name):
        norm_scale = tf.get_variable("norm_scale", [num_tokens, d_model], ones)
        normed = apply_rms_norm(x, norm_scale)
        gate_up_kernels = _independent_kernels("gate_up_kernel", num_tokens, d_model, 2*hidden_dim)
        gate_up_bias    = tf.get_variable("gate_up_bias", [num_tokens, 2*hidden_dim], zeros)
        gate_up = _token_independent_matmul(normed, gate_up_kernels) + gate_up_bias[None]
        gate, up = tf.split(gate_up, 2, axis=-1)
        middle = tf.nn.silu(gate) * up
        down_kernels = _independent_kernels("down_kernel", num_tokens, hidden_dim, d_model)
        down_bias    = tf.get_variable("down_bias", [num_tokens, d_model], zeros)
        output = _token_independent_matmul(middle, down_kernels) + down_bias[None]
        return x + output if residual else output
```

⚠️ `rankmixer_v2_layers.py` 顶层 `import ego`（`_token_independent_matmul` 需要 `ego.is_training_mode()` 分支），所以 **import `tim_block` 就需要 EGO 环境**。`fim_block` 不用这个文件（它自带同类实现）。
⚠️ `apply_rms_norm` 现在唯一的消费者是 `tim_block`，所以它的实现细节只影响 TIM。

#### 5.11.3 变量清单（scope `brv4_tim_parallel`）

每个 expert（scope `expert_0` / `expert_1`）：

| 变量名 | shape | 初始化 | 说明 |
|---|---|---|---|
| `query_norm_scale` | `[7,128]` | ones | per-query RMSNorm 增益 |
| `query_kernel_0..6` | 7 × `[128,128]` | glorot_uniform | **per-query 独立** Wq |
| `memory_norm_scale` | `[16,128]` | ones | 行 0-7 归 user token、8-15 归 item token |
| `user_kv_kernel_0..7` | 8 × `[128,256]` | glorot_uniform | per-memory-token Wkv |
| `item_kv_kernel_0..7` | 8 × `[128,256]` | glorot_uniform | 同上 |
| `output_kernel` | `[128,128]` | glorot_uniform | **expert 内 7 个 query 共用** |
| `output_bias` | `[128]` | zeros | 同上 |

scope 根下：`query_seed [1,7,128]`（glorot_uniform）。
scope `fusion`：`kernel_0..6` = 7 × `[256,128]`（glorot_uniform）、`bias [7,128]`（zeros）。
scope `task_ffn`：`norm_scale [7,128]`、`gate_up_kernel_0..6` = 7×`[128,256]`、`gate_up_bias [7,256]`、`down_kernel_0..6` = 7×`[128,128]`、`down_bias [7,128]`。

⚠️ **Q/K/V 均无 bias**（历史上曾有 `query_bias`/`user_kv_bias`/`item_kv_bias`，已删除）。`output_bias`、`fusion/bias`、`task_ffn` 的两个 bias **保留**。
💡 去掉 `q_bias` 是无损的：`W_t`、`seed_t`、`q_norm_t` 三者都 per-query 独立可学，`W_t · n_t` 已能张成 `R^128` 任意向量。去掉 K/V bias 则**有表达力损失**（失去了一个与输入无关的 (query, memory token) 对偶 logit 偏置，类似可学习的 attention prior），是刻意的取舍。

单 expert **658,432** 参数，TIM 合计 **1,895,680**。

#### 5.11.4 `_forward` 完整流程

```python
def _forward(self, user_tokens, item_tokens, training):
    seed = tf.get_variable("query_seed", [1, 7, 128], initializer=tf.glorot_uniform_initializer())
    experts = self._create_experts()
    fusion  = self._create_fusion()

    # 每个 expert 的投影保持独立 rank-2 变量；输出 stack 到 expert 轴，
    # 使下面的 attention 对两个 expert 只跑一次。
    queries = tf.stack([self._project_query(seed, e)          for e in experts], axis=1)  # [1,2,7,128]
    user_kv = tf.stack([self._project_user_kv(user_tokens, e) for e in experts], axis=1)  # [B_u,2,8,256]
    item_kv = tf.stack([self._project_item_kv(item_tokens, e) for e in experts], axis=1)  # [B_i,2,8,256]

    if training:
        context = self._attend_experts(queries, user_kv, item_kv)
    else:
        context = request_shared_tim_attention(queries, user_kv, item_kv, self.num_heads)

    if training:
        # 每 expert 一个 rank-2 Dot，保持已验证的 XLA 训练下降；
        # 把 rank-3 stack 后的 kernel 放进 batched matmul 会触发 TransformMmt4D。
        branch = tf.stack([
            _shared_linear(context[:, index], expert["o_kernel"], expert["o_bias"],
                           name="expert_{}_output".format(index))
            for index, expert in enumerate(experts)], axis=1)                      # [B,2,7,128]
    else:
        branch = _expert_linear(
            context, tf.stack([e["o_kernel"] for e in experts], axis=0)
        ) + tf.stack([e["o_bias"] for e in experts], axis=0)[None, :, None, :]

    merged = tf.reshape(tf.transpose(branch, [0, 2, 1, 3]),
                        [tf.shape(branch)[0], 7, EXPERT_COUNT * 128],
                        name="expert_concat")                                     # [B,7,256]
    fused = _token_independent_matmul(merged, fusion["kernel"]) \
          + tf.expand_dims(fusion["bias"], axis=0)                                # [B,7,128]
    hidden = seed + fused                        # ⚠️ seed 全程只加这一次
    return packed_per_token_swiglu(hidden, 7, 128, 128, "task_ffn")               # [B,7,128]
```

其中：

```python
def _project_query(self, seed, expert):
    normalized = apply_rms_norm(seed, expert["q_norm"])
    return _token_independent_matmul(normalized, expert["q_kernel"])      # [1,7,128]

def _project_user_kv(self, user_tokens, expert):
    return _token_independent_matmul(
        apply_rms_norm(user_tokens, expert["memory_norm"][:8]), expert["user_kv_kernel"])

def _project_item_kv(self, item_tokens, expert):
    return _token_independent_matmul(
        apply_rms_norm(item_tokens, expert["memory_norm"][8:]), expert["item_kv_kernel"])

def _split_query_heads(self, queries):   # [B,E,T,D] -> [B,E,H,T,depth]
    return tf.transpose(tf.reshape(queries, [tf.shape(queries)[0], E, 7, 8, 16]), [0,1,3,2,4])

def _split_kv_heads(self, kv):           # [B,E,J,2D] -> K,V 各 [B,E,H,J,depth]
    key, value = tf.split(kv, 2, axis=-1)
    return [tf.transpose(tf.reshape(p, [tf.shape(kv)[0], E, J, 8, 16]), [0,1,3,2,4])
            for p in (key, value)]

def _merge_context_heads(self, context): # [B,E,H,T,depth] -> [B,E,T,D]
    merged = tf.transpose(context, [0,1,3,2,4])
    return tf.reshape(merged, [tf.shape(merged)[0], EXPERT_COUNT, 7, 128])
```

#### 5.11.5 两条 attention 路径（⚠️ 语义必须等价）

**training：`_attend_experts(queries, user_kv, item_kv)`**

```python
query = self._split_query_heads(queries)                       # [1,E,H,T,16]
user_key, user_value = self._split_kv_heads(user_kv)           # [B,E,H,8,16]
item_key, item_value = self._split_kv_heads(item_kv)
user_logits = tf.matmul(query, user_key, transpose_b=True) * self.scale   # [B,E,H,T,8]
item_logits = tf.matmul(query, item_key, transpose_b=True) * self.scale   # [B,E,H,T,8]
logits  = tf.concat([user_logits, item_logits], axis=-1, name="memory_logits_16")
weights = tf.nn.softmax(logits, axis=-1, name="weights")
context = (tf.matmul(weights[..., :8], user_value)
         + tf.matmul(weights[..., 8:], item_value))
return self._merge_context_heads(context)
```

**serving：`request_shared_tim_attention(query, user_kv, item_kv, num_heads)`**

```python
_, experts, tasks, width = query.shape.as_list()      # [1,E,T,D]，全静态
requests = tf.shape(user_kv)[0]                        # R
with tf.name_scope("brv4_request_tim_attention") as scope:
    tf.compat.v1.add_to_collection(_ATTENTION_SCOPES, scope)     # ⚠️ 编译保护
    q = tf.transpose(tf.reshape(tf.cast(query, tf.float32), [1,E,T,H,depth]), [0,1,3,2,4])

    def partition(kv, name):
        tokens = kv.shape.as_list()[2]
        key, value = tf.split(tf.cast(kv, tf.float32), 2, axis=-1)
        shape = [-1, experts, tokens, num_heads, depth]
        key   = tf.identity(tf.transpose(tf.reshape(key,   shape), [0,1,3,2,4]), name="key")
        value = tf.identity(tf.transpose(tf.reshape(value, shape), [0,1,3,2,4]), name="value")
        logits      = tf.matmul(q, key, transpose_b=True) * (depth ** -0.5)
        maximum     = tf.reduce_max(logits, axis=-1, keepdims=True, name="maximum")
        exponent    = tf.exp(logits - maximum)
        denominator = tf.reduce_sum(exponent, axis=-1, keepdims=True, name="denominator")
        numerator   = tf.matmul(exponent, value, name="numerator")
        return maximum, denominator, numerator

    user_max, user_den, user_num = partition(user_kv, "user")     # rows = R
    item_max, item_den, item_num = partition(item_kv, "item")     # rows = C*R
    scalar_shape = [-1, requests, experts, num_heads, tasks, 1]
    item_max = tf.reshape(item_max, scalar_shape)
    item_den = tf.reshape(item_den, scalar_shape)
    item_num = tf.reshape(item_num, [-1, requests, experts, num_heads, tasks, depth])
    maximum    = tf.maximum(user_max[None], item_max)
    user_scale = tf.exp(user_max[None] - maximum)
    item_scale = tf.exp(item_max - maximum)
    numerator   = user_scale * user_num[None] + item_scale * item_num
    denominator = user_scale * user_den[None] + item_scale * item_den
    context = numerator / denominator
    context = tf.transpose(context, [0,1,2,4,3,5])
    context = tf.reshape(context, [-1, experts, tasks, width], name="context")
    return tf.cast(context, query.dtype)
```

**等价性**：两条路径都是对 16 个 memory token 的**一次联合 softmax**。serving 用的是 log-sum-exp 分区归并恒等式：

\[\sum_{j\in A} e^{l_j-M}v_j = e^{M_A-M}\cdot\underbrace{\sum_{j\in A}e^{l_j-M_A}v_j}_{num_A}\]

因为 `exp(M_A-M)` 对分区 A 内所有 j 是同一标量，可提到求和号外。`num_A` 只依赖 user 侧的量，所以能在 batch=R 上算完。

⚠️ **不需 epsilon 的严格理由**：`den_A >= exp(l_argmax - M_A) = 1`，同理 `den_B >= 1`；且 `M_A`/`M_B` 必有一个等于 `M`，对应 scale 恰为 1。故 `denominator >= 1`，既不会除零也不会双侧同时下溢。前提是**两个分区的 token 数固定为 8 和 8**（TIM memory 是 FIM 输出 token，无 padding）。若将来给 TIM 引入可变长 memory，epsilon 必须加回来。
⚠️ **不能把归并“优化”成两次独立 softmax 再加权**。`tests/tim_numpy_contracts.py::test_joint_softmax_not_two_independent_softmaxes` 与 `tests/tim_path_contracts.py` 的同名用例对此有守卫（差异 > 1e-3）。
💡 两条路径的实测差异：fp64 worst **7.275e-16**（机器精度，证明数学恒等）；fp32 worst **4.881e-07**（≈4 ULP），真实量级 **2.5e-07**。多请求隔离 R=2/3/4/5 零泄漏。详见 `new_optimize.md` 第一部分。
⚠️ **不要把 training 也改成 serving 写法**：训练时 `B_u == B_i`，无可提量，反而多 ~12 个算子、打散 XLA 的 softmax 融合，并引入 `tf.maximum` 平局时梯度归属的版本相关风险。

### 5.12 场景硬路由、任务塔、loss 与输出协议

#### 5.12.1 `_scene_condition` 与 `scene_hidden`

```python
def _scene_condition(self):
    onehot = self.scene_onehot_input                        # ⚠️ 原始 clip 后的，不是归一化过的 token
    pp  = tf.minimum(tf.reduce_sum(onehot[:, :8], axis=-1, keepdims=True), 1.0)
    pdp = onehot[:, 8:9]
    dd  = onehot[:, 9:10]
    return tf.concat([dd, pdp, pp], axis=-1, name="brv4_scene_dd_pdp_pp")     # [B,3]

# build_esmm 内：
hidden = dict(zip(self.TIM_TOKEN_NAMES, tf.unstack(tim_tokens, axis=1)))       # 7 × [B,128]
scene_hidden = (scene_condition[:, 0:1] * hidden["dd"]
              + scene_condition[:, 1:2] * hidden["pdp"]
              + scene_condition[:, 2:3] * hidden["pp"])                        # [B,128]
scene_hidden = tf.identity(scene_hidden, name="brv4_selected_scene_token")
gate_input = tf.concat([scene_hidden, hidden["common"], scene_condition], axis=-1,
                       name="brv4_tower_gate_input")                           # [B,259]
```

⚠️ **这是硬路由，不是软注意力**：`scene_condition` 三列中恰有一列为 1（dd/pdp/pp 互斥），所以 `scene_hidden` 实际选中一个场景 token。一旦对 `scene_onehot` 减均值，三列不再是 0/1，硬选择会退化成任意线性组合。
⚠️ `pp` 是前 8 位的求和再 `minimum(·, 1.0)`，不是单个 slot。

#### 5.12.2 tower_input 与路由

```python
routes = {
    "click":       ("click", "common"),
    "atc":         ("atc", "click", "common"),
    "order":       ("order", "click", "atc", "common"),
    "place_order": ("order", "click", "atc", "common"),
    "ads_order":   ("order", "click", "atc", "common"),
}
# 断言：routes 里引用的 token 必须都在 TIM_TOKEN_NAMES 里
missing = sorted({t for ts in routes.values() for t in ts} - set(self.TIM_TOKEN_NAMES))
if missing: raise KeyError("task routes reference missing TIM tokens: {}".format(missing))

tower_inputs = [tf.concat([rank_global] + [hidden[t] for t in routes[target]] + [scene_hidden],
                          axis=-1, name="brv4_{}_tower_input".format(target))
                for target in self.target_names]
```

| 任务 | 路由 token 数 | tower_input 宽度 |
|---|---:|---:|
| click | 2 | 256 + 2×128 + 128 = **640** |
| atc | 3 | 256 + 3×128 + 128 = **768** |
| order / place_order / ads_order | 4 | 256 + 4×128 + 128 = **896** |

⚠️ 拼接顺序固定为 `rank_global → 路由内 TIM token（按 routes 元组顺序）→ scene_hidden`。
💡 三个 order 类任务的 `tower_input` 内容完全相同，但因 `name` 不同不会被 CSE 合并（已知冗余，见 `optimize.md` S2）。

#### 5.12.3 `_task_towers(tower_inputs, gate_input, names)`

`names = ["p0_{target}_brv4_tower" for target in target_names]`，每塔 7 层。所有层用 grouped MatMul 执行，但**参数逐塔独立**。

```python
TASK_TOWER_INPUT_DIM = 896
TASK_GATE_INPUT_DIM  = 260

# 1) 输入右侧零填充到统一宽度，再 stack 成 [B,5,896]
padded_inputs = [tf.pad(x, [[0,0],[0, 896 - x.shape[-1]]]) for x in tower_inputs]
packed_inputs = tf.stack(padded_inputs, axis=1, name="brv4_grouped_task_inputs")
padded_gate_input = tf.pad(gate_input, [[0,0],[0, 260 - gate_input.shape[-1]]],
                           name="brv4_grouped_gate_input")            # 259 -> 260

# 2) 每塔在自己的 variable_scope 下创建 7 组 (kernel, bias)
for name, input_dim in zip(names, input_dims):
    with tf.variable_scope(name):
        hidden_0      : _task_dense_parameters(input_dim, 128, "hidden_0")
        gate_0_hidden : _task_dense_parameters(259,        64, "gate_0_hidden")   # ⚠️ 用原始 259
        gate_0        : _task_dense_parameters(64,        128, "gate_0")
        hidden_1      : _task_dense_parameters(128,        64, "hidden_1")
        gate_1_hidden : _task_dense_parameters(259,        64, "gate_1_hidden")
        gate_1        : _task_dense_parameters(64,         64, "gate_1")
        output        : _task_dense_parameters(64,          1, "output")

# 3) 前向
hidden        = leaky_relu(_grouped_task_linear(packed_inputs,     hidden0_params,     "brv4_grouped_hidden_0"))
gate0_hidden  = leaky_relu(_shared_task_linear (padded_gate_input, gate0_hidden_params, "brv4_grouped_gate_0_hidden"))
gate0         = 2.0 * sigmoid(_grouped_task_linear(gate0_hidden,   gate0_params,       "brv4_grouped_gate_0"))
hidden       *= gate0
hidden        = leaky_relu(_grouped_task_linear(hidden,            hidden1_params,     "brv4_grouped_hidden_1"))
gate1_hidden  = leaky_relu(_shared_task_linear (padded_gate_input, gate1_hidden_params, "brv4_grouped_gate_1_hidden"))
gate1         = 2.0 * sigmoid(_grouped_task_linear(gate1_hidden,   gate1_params,       "brv4_grouped_gate_1"))
hidden       *= gate1
return sigmoid(_grouped_task_linear(hidden, output_params, "brv4_grouped_output"))     # [B,5,1]
```

三个辅助方法：

```python
@staticmethod
def _task_dense_parameters(input_dim, output_dim, name):
    with tf.variable_scope(name):
        kernel = tf.get_variable("kernel", [input_dim, output_dim], glorot_uniform)
        bias   = tf.get_variable("bias",   [output_dim],           zeros)
    return kernel, bias

@staticmethod
def _pad_task_kernel(kernel, target_width):
    """[in,O] -> [target_width,O]，底部补零。in > target_width 则报错。"""
    if input_width == target_width: return kernel
    return tf.pad(kernel, [[0, target_width - input_width], [0, 0]])

def _grouped_task_linear(self, inputs, parameters, name):
    """[B,N,D] x N 个独立 [D,O] -> [B,N,O]，一个 batched MatMul。"""
    kernels = [self._pad_task_kernel(k, inputs.shape[2]) for k, _ in parameters]
    with tf.name_scope(name):
        group_major = tf.transpose(inputs, [1, 0, 2])                       # [N,B,D]
        output = tf.matmul(group_major, tf.stack(kernels, axis=0))          # [N,B,O]
        output += tf.expand_dims(tf.stack([b for _, b in parameters], axis=0), axis=1)
        return tf.transpose(output, [1, 0, 2])                              # [B,N,O]

def _shared_task_linear(self, inputs, parameters, name):
    """一份共享 [B,D] 输入 + N 个独立 kernel -> [B,N,O]，一个 rank-2 MatMul。"""
    packed_kernel = tf.concat([self._pad_task_kernel(k, D_in) for k, _ in parameters], axis=1)
    packed_bias   = tf.concat([b for _, b in parameters], axis=0)
    output = tf.matmul(inputs, packed_kernel) + packed_bias                 # [B, N*O]
    return tf.reshape(output, [-1, len(parameters), output_dim])
```

⚠️ **`gate_0_hidden` / `gate_1_hidden` 的 kernel 是按原始 `gate_dim=259` 创建的**，然后在 `_shared_task_linear` 里被 `_pad_task_kernel` 补到 260 以匹配已填充的输入。所以变量 shape 是 `[259,64]`，不是 `[260,64]`。
⚠️ **零填充的数学依据**：`[x, 0] @ [[W], [W_pad]] + b = x @ W + b`。补零行对应的输入恒为 0，所以不增加前向容量。但那些 kernel 行会收到**恒零梯度**，永远停在初始值（`param_regularizer=None`，无正则惩罚；Adam 在零梯度下更新为 0，不漂移）。
⚠️ **门控是 `2.0 * sigmoid(·)`**，值域 (0,2)，不是标准 sigmoid。两层门控分别乘在 `hidden_0` 与 `hidden_1` 的输出上。
💡 `_shared_task_linear` 用于两个 gate 的第一层，因为它们读同一份 `padded_gate_input`；`_grouped_task_linear` 用于输入逐塔不同的层。

#### 5.12.4 预测、label、loss、target

```python
grouped_predictions = self._task_towers(tower_inputs, gate_input, names)     # [B,5,1]
predictions = [tf.identity(p, "p0_{}".format(target))
               for target, p in zip(self.target_names,
                                    tf.unstack(grouped_predictions, axis=1))]
labels, weights, losses = [], [], []
for idx, prediction in enumerate(predictions):
    label, weight = ego.get_label_weight(target_name="label_0_{}".format(idx * 2),
                                         label_idx=idx * 2)
    labels.append(label); weights.append(weight)
    losses.append(cross_entropy_loss(label, prediction))
self._set_outputs(predictions, labels, weights, losses, cache_norm)
```

⚠️ **`label_idx = idx * 2`** —— 5 个任务取 label 向量的第 0/2/4/6/8 位，`target_name` 是 `label_0_0` / `label_0_2` / `label_0_4` / `label_0_6` / `label_0_8`。

```python
def cross_entropy_loss(label, prediction, epsilon=1e-6):
    return (-label * tf.math.log(prediction + epsilon)
            -(1.0 - label) * tf.math.log(1.0 - prediction + epsilon))
```

⚠️ **这是带 epsilon 的概率域 BCE，不是 `sigmoid_cross_entropy_with_logits`**。两者在 `epsilon > 0` 时**不逐值等价**（`y=1` 时前者导数是 `-p(1-p)/(p+eps)`，后者是 `p-1`）。改成 logits 形式属于刻意的数值行为变更，需单独开关与训练曲线对比，**不要当作等价优化顺手改掉**。

```python
def _set_outputs(self, predictions, labels, weights, losses, cache_norm):
    names = ["p0_" + t for t in self.target_names]
    phase = self._phases[0]
    phase["losses"], phase["loss_weights"] = losses, weights
    phase["target_names"], phase["targets"], phase["labels"] = names, predictions, labels
    phase["predict_target_names"]  = names + ["item_cache_norm"]
    phase["predict_targets"]       = predictions + [cache_norm]
    phase["predict_labels"]        = labels + [labels[0]]
    phase["predict_weights"]       = weights + [weights[0]]
    phase["debug_target_names"]    = self.debug_nodes          # 当前为空列表

def build_loss(self):
    self._phases[0]["phase_loss"] = -sum(
        tf.reduce_sum(loss * weight) for loss, weight in zip(losses, loss_weights))

def build_monitoring(self):
    self._phases[0]["scalar_tensor"] = self.scalar_tensor      # 当前为空列表

def format_ego_targets(self):
    for name, prediction, label, weight, loss in zip(...):
        ego.Target(name, tf.identity(prediction, name), label, weight, loss, ego.MetricType.GAUC)
    for name, prediction, label, weight in zip(predict_*):
        if name not in phase["target_names"]:                 # 即 item_cache_norm
            ego.Target(name, tf.identity(prediction, name), label, weight, None, ego.MetricType.GAUC)
```

⚠️ **loss 前面有负号**：`cross_entropy_loss` 返回的是负对数似然的**正值**形式（`-y·log p - (1-y)·log(1-p)`），而 `phase_loss = -Σ reduce_sum(loss*weight)`。两个负号抵消，最终是最小化交叉熵。重写时不要把其中一个负号弄丢。
⚠️ `item_cache_norm` 用 `labels[0]` / `weights[0]`（click 的）作为占位，`loss=None`，仅作为预测/监控输出。

#### 5.12.5 入口的 Round 与编译

```python
p1 = model.build_graph()[0]
# ... 写 export.yaml ...
if ego.is_training_mode():
    eval_round  = ego.OfflineRound(name="eval",
        targets=p1["predict_target_names"] + p1.get("debug_target_names", []),
        dump_scalar_tensors_to_tensorboard=p1["scalar_tensor"], final_loss=None)
    train_round = ego.OfflineRound(name="p1", targets=p1["target_names"],
        final_loss=p1["phase_loss"], train_sparse=True)
    ego.compile(rounds=[eval_round, train_round])
else:
    online_round = ego.OnlineRound(name="online", targets=p1["predict_target_names"])
    compile_with_request_attention(rounds=[online_round])      # ⚠️ 不是 ego.compile
```

**`compile_with_request_attention(rounds)`**：

```python
scopes = tuple(tf.compat.v1.get_collection(_ATTENTION_SCOPES))
if not scopes: return ego.compile(rounds=rounds)
compiler = importlib.import_module("ego.tensorflow.training.compiler")
original_editor = compiler.GraphEditor

class RequestAttentionGraphEditor(original_editor):
    def fit_batch_size(self, shared_input_tensor, unshared_input_tensor, output_node, idx):
        if output_node.name.startswith(scopes): return          # 跳过 re-tile
        return super().fit_batch_size(shared_input_tensor, unshared_input_tensor, output_node, idx)

try:
    compiler.GraphEditor = RequestAttentionGraphEditor
    return ego.compile(rounds=rounds)
finally:
    compiler.GraphEditor = original_editor                     # ⚠️ 必须恢复
```

⚠️ 这是对 EGO 内部 API 的 monkey-patch，EGO 升级可能失效；`try/finally` 保证即使编译失败也恢复原类。它**不修改已安装的 EGO 文件**。
⚠️ `_ATTENTION_SCOPES = "uniformer_request_attention_scopes"`（名字沿用参考实现，不要改）。注册进这个集合的 scope 有三个：`uniformer_request_attention`（FIM）、`brv4_request_tim_attention`（TIM）、`brv4_short_cosines` / `brv4_long_cosines`（SimTier）。
⚠️ **训练态不调这个适配器**（训练走普通 `ego.compile`），因为训练时没有请求/候选不对称。

---

## 6. 参数清单

全部数字由 `mcconf_brnew_v4.yaml` 的组维度 + 源码声明的 shape 推导，可用 `print_total_params()` 交叉验证。

💡 **可复现**：`python3 tests/param_inventory.py` 会重新从 yaml + shape 公式算出下面所有数字，并在末尾断言它们与本节完全一致（不一致则 `AssertionError`）。纯 NumPy + PyYAML，无需 TensorFlow/EGO。改了特征配置或模块 shape 后跑一次，就能知道参数总量漂移了多少。

### 6.1 汇总

| 模块 | 可训练 | 非训练（moving stats） | 占比 |
|---|---:|---:|---:|
| user token 投影（8 个） | 2,853,888 | 14,856 | 24.95% |
| 序列流编码（5 条） | 110,976 | 0 | 0.97% |
| item token（含 teacher） | 750,336 | 1,792 | 6.56% |
| **FIM × 2 层** | **4,637,184** | 0 | **40.54%** |
| rank_global | 393,984 | 4,096 | 3.44% |
| **TIM** | **1,895,680** | 0 | **16.57%** |
| 任务塔（5 塔） | 795,333 | 0 | 6.95% |
| **总计** | **11,437,381** | **20,744** | 100% |

⚠️ `brv4_item_cache_teacher`（522,880）**仅训练态存在**，serving 图里没有。所以 **serving 可训练参数 = 10,914,501**。

### 6.2 user token 投影

| 变量 scope | 输入维 | kernel | bias | 小计 | input_norm moving |
|---|---:|---:|---:|---:|---:|
| `brv4_user_token` | 272 | 34,816 | 128 | 34,944 | 544 |
| `brv4_user_short_token` | 904 | 115,712 | 128 | 115,840 | 1,808 |
| `brv4_user_long_token` | 1656 | 211,968 | 128 | 212,096 | 3,312 |
| `brv4_context_token` | 560 | 71,680 | 128 | 71,808 | 1,120 |
| `brv4_scene_token` | 10 | 1,280 | 128 | 1,408 | 20 |
| `brv4_click_seq_seed` | 176 | 22,528 | 128 | 22,656 | 352 |
| `brv4_order_seq_seed` | 136 | 17,408 | 128 | 17,536 | 272 |
| `brv4_user_global_token` | 3714 → **640** | 2,376,960 | 640 | **2,377,600** | 7,428 |

💡 `brv4_user_global_token` 一个就占 user 侧的 **83%**，是全模型最大的单个 GEMM。它的 640 维输出切成 5 个 chunk，只有 chunk 0 用作 user token，chunk 1-4 给 item 侧做 fusion query。

### 6.3 序列流（scope `brv4_sequence_{group}`）

| scope | projection | position | type | 小计 |
|---|---:|---:|---:|---:|
| `..._user_click_seq` | 176×128+128 = 22,656 | 50×128 = 6,400 | 128 | 29,184 |
| `..._user_order_seq` | 136×128+128 = 17,536 | 6,400 | 128 | 24,064 |
| `..._user_click_long_seq` | 80×128+128 = 10,368 | 6,400 | 128 | 16,896 |
| `..._user_cart_seq` | 112×128+128 = 14,464 | 6,400 | 128 | 20,992 |
| `..._upstream_impression_seq` | 128×128+128 = 16,512 | 25×128 = 3,200 | 128 | 19,840 |
| **合计** | | | | **110,976** |

`atc_generic` 不新增参数（它是在已投影的 atc token 上池化得到的）。

### 6.4 item token

| scope | shape | 参数量 | 备注 |
|---|---|---:|---|
| `brv4_item_cache_teacher/layer_1` | `[816,640]`+`[640]` | 522,880 | 仅训练；`norms=[True]` 另有 1,632 moving |
| `brv4_recall_token/layer_1` | `[80,128]`+`[128]` | 10,368 | `input_norm`，160 moving |
| `brv4_img_token/layer_1` | `[231,128]`+`[128]` | 29,696 | 无 norm |
| `brv4_title_token/layer_1` | `[435,128]`+`[128]` | 55,808 | 无 norm |
| `brv4_fusion_queries/{route}_query` ×4 | `[256,128]`+`[128]` | 4×32,896 = 131,584 | route ∈ FUSION_ROUTES |

item token 0（`item_token`）无参数 —— 它直接是 cache 的 chunk 0。

### 6.5 FIM（每层，scope `brv4_fim_{layer_idx}`）

| 变量 | shape | 参数量 |
|---|---|---:|
| `route_kv_norm_scale` | `[6,128]` | 768 |
| `route_kv_kernel` | `[6,128,256]` | 196,608 |
| `route_kv_bias` | `[6,256]` | 1,536 |
| `q_norm_scale` | `[16,128]` | 2,048 |
| `q_kernel` | `[16,128,128]` | 262,144 |
| `o_kernel` | `[16,128,128]` | 262,144 |
| `o_bias` | `[16,128]` | 2,048 |
| `s_norm_scale` / `mix_norm_scale` / `ffn_norm_scale` | 各 `[16,128]` | 3×2,048 = 6,144 |
| `s_ffn/gate_up` | `[16,128,256]` | 524,288 |
| `s_ffn/gate_up_bias` | `[16,256]` | 4,096 |
| `s_ffn/down` | `[16,128,128]` | 262,144 |
| `s_ffn/down_bias` | `[16,128]` | 2,048 |
| `m_ffn/*` | 同 s_ffn | 792,576 |
| **单层小计** | | **2,318,592** |
| **× 2 层** | | **4,637,184** |

`UIHeadMixer` 无参数（两个 `tf.constant` 索引表，1024 + 512 个 int32）。

### 6.6 rank_global

| scope | shape | 参数量 |
|---|---|---:|
| `brv4_rank_user_a/layer_1` | `[1024,128]`+`[128]` | 131,200（+2,048 moving） |
| `brv4_rank_item_b/layer_1` | `[1024,128]`+`[128]` | 131,200（+2,048 moving） |
| `brv4_rank_global_mlp/layer_1` | `[256,256]`+`[256]` | 65,792 |
| `brv4_rank_global_mlp/layer_2` | `[256,256]`+`[256]` | 65,792 |
| **小计** | | **393,984** |

### 6.7 TIM（scope `brv4_tim_parallel`）

| 变量 | shape | 参数量 |
|---|---|---:|
| `expert_{0,1}/query_norm_scale` | `[7,128]` ×2 | 1,792 |
| `expert_{0,1}/query_kernel_0..6` | 7×`[128,128]` ×2 | 229,376 |
| `expert_{0,1}/memory_norm_scale` | `[16,128]` ×2 | 4,096 |
| `expert_{0,1}/user_kv_kernel_0..7` | 8×`[128,256]` ×2 | 524,288 |
| `expert_{0,1}/item_kv_kernel_0..7` | 8×`[128,256]` ×2 | 524,288 |
| `expert_{0,1}/output_kernel` | `[128,128]` ×2 | 32,768 |
| `expert_{0,1}/output_bias` | `[128]` ×2 | 256 |
| *单 expert 小计* | | *658,432* |
| `query_seed` | `[1,7,128]` | 896 |
| `fusion/kernel_0..6` | 7×`[256,128]` | 229,376 |
| `fusion/bias` | `[7,128]` | 896 |
| `task_ffn/norm_scale` | `[7,128]` | 896 |
| `task_ffn/gate_up_kernel_0..6` | 7×`[128,256]` | 229,376 |
| `task_ffn/gate_up_bias` | `[7,256]` | 1,792 |
| `task_ffn/down_kernel_0..6` | 7×`[128,128]` | 114,688 |
| `task_ffn/down_bias` | `[7,128]` | 896 |
| **总计** | | **1,895,680** |

其中 per-memory-token 的 K/V 矩阵占 **1,048,576**（核心矩阵的 62%）。

### 6.8 任务塔（scope `p0_{target}_brv4_tower`）

| 层 | kernel shape（每塔） | 5 塔合计 |
|---|---|---:|
| `hidden_0` | `[640|768|896, 128]` | 524,928 |
| `gate_0_hidden` | `[259, 64]` | 83,200 |
| `gate_0` | `[64, 128]` | 41,600 |
| `hidden_1` | `[128, 64]` | 41,280 |
| `gate_1_hidden` | `[259, 64]` | 83,200 |
| `gate_1` | `[64, 64]` | 20,800 |
| `output` | `[64, 1]` | 325 |
| **小计** | | **795,333** |

`hidden_0` 的 524,928 = `(640×128+128) + (768×128+128) + 3×(896×128+128)`。

---

## 7. 训练 / serving 差异总表

| 环节 | 训练 | serving |
|---|---|---|
| batch 语义 | 所有特征 `B_train`，user/item 同 batch | user = `R`（COMMON），item = `C·R`（ITEM），行序 `[candidate, request]` |
| item embedding | `DenseTower(816→640)` teacher + `replace_gradient`，写入 slot 30102 | **只查表** `assign_slot_emb`，816 维 item 特征完全不进图 |
| `brv4_item_cache_teacher` | 存在（522,880 参数） | **不存在** |
| FIM `_item_ca` | `broadcast_memory=False` → `masked_attention` | `broadcast_memory=True` → `request_shared_attention` |
| FIM `_user_ca` | `broadcast_memory=False`（默认） | 同（user 本来就是请求粒度） |
| TIM attention | `_attend_experts`：分离 logits → concat → 一次联合 softmax | `request_shared_tim_attention`：分区 online-softmax 归并 |
| TIM O 投影 | 逐 expert `_shared_linear`（rank-2 Dot，避开 XLA） | `_expert_linear`（stack 后一个 batched matmul） |
| `_token_independent_matmul` | N 个独立 rank-2 MatMul + Concat | `tf.stack` 后一个 batched matmul |
| SimTier `_histogram_counts` | `unsorted_segment_sum`（线性时间） | 8-bin 分块 Equal（TRT 可解析） |
| 编译 | `ego.compile(rounds=[eval_round, train_round])` | `compile_with_request_attention(rounds=[online_round])` |
| Round | `OfflineRound("eval")` + `OfflineRound("p1", train_sparse=True)` | `OnlineRound("online")` |
| `INormalization` | 前向用 moving stats，batch 统计只用于 EMA 更新 | 前向用 moving stats（**无 mode 分支**） |

⚠️ **`INormalization` 不是标准 Keras BatchNormalization**：它的 `call` 没有 `is_serving_mode()` 分支，训练与推理前向**都用 moving stats**。（EGO 另有一个 `IBatchNormalization` 才有 mode 分支，`DenseTower(norms=[True])` 用的是前者。）

---

## 8. 必须保持的不变量（建图期断言清单）

重写时应把这些全部实现为 `raise`，而不是依赖运行时才发现：

| # | 不变量 | 位置 |
|---|---|---|
| 1 | `len(used_slot_map) == len(sparse_input)` | `build_inputs` |
| 2 | 必需特征组全部存在（USER/CONTEXT/ITEM/RECALL/QUERY/IMAGE_SEQ/TITLE_SEQ/LONG_IMAGE_SEQ + 5 个 SEQUENCE_SPECS 源组） | `_check_feature_groups` |
| 3 | 序列组内**全是** tile_nf 元组（混入静态特征则 raise） | `build_group_input` |
| 4 | 序列组内所有特征的 padded length 相同 | `build_group_input` |
| 5 | mcconf `item_cache_slot` 必须是 slot 30102 且 dim=640 | 入口文件 |
| 6 | SEQUENCE_SPECS 的 5 条断言（route 不重、`stream<=end-start`、`(end-start)%stream==0`、`stream%generic==0`、FUSION_ROUTES 均有编码流） | `_check_sequence_specs` |
| 7 | `_pool_fixed`：`source_length % output_tokens == 0`，且两者静态可知 | `_pool_fixed` |
| 8 | `user_tokens.shape[1] == 8` / `item_tokens.shape[1] == 8` | `_build_user_tokens` / `_build_item_tokens` |
| 9 | `user_global` 宽度 == 640；`cache_emb` 宽度 == 640 | `_require_width` |
| 10 | SimTier 输出宽度 == 231 / 231 / 204 | `_require_width` |
| 11 | FIM：`d_model % num_heads == 0`；user/item token 静态是 `[B,8,128]`；每个 sequence view 是 `[B,ROUTE_LENGTHS[r],128]` 且 mask 是 `[B,ROUTE_LENGTHS[r]]`；`hidden_dim > 0` | `FIMBlock` |
| 12 | `UIHeadMixer`：`num_tokens == 2*user_tokens` 且 `d_model % num_tokens == 0` | `UIHeadMixer.__init__` |
| 13 | `GENERIC_MEMORY` 偏移量与 `ROUTE_LENGTHS` 一致（`check_memory_layout` 类校验） | FIM 契约 |
| 14 | TIM：`d_model % num_heads == 0`；两侧输入静态是 `[B,8,128]`；`_expert_linear` 要求 expert/token/width 全静态 | `ParallelTIMBlock` |
| 15 | `request_shared_tim_attention`：expert/task/width 静态且 `width % num_heads == 0` | helper |
| 16 | `request_cosines`：keys 的 group/length/width 静态 | helper |
| 17 | SimTier：短序列必须 6 路、长序列必须 2 路；tiers 为正；`fine % coarse == 0`；length 静态 | `short_simtier` / `long_simtier` / `_coarse_from_fine` |
| 18 | `routes` 引用的 TIM token 必须都在 `TIM_TOKEN_NAMES` 里 | `build_esmm` |
| 19 | `len(tower_inputs) == len(names)`；每塔输入宽 ≤ 896；`gate_dim` ≤ 260 且静态 | `_task_towers` |
| 20 | `_pad_task_kernel`：`input_width <= target_width` | `_pad_task_kernel` |
| 21 | `_shared_task_linear`：所有 kernel 的输出宽相同且静态 | `_shared_task_linear` |
| 22 | `_grouped_task_linear`：输入是 rank-3 且 `shape[1] == len(parameters)` | `_grouped_task_linear` |
| 23 | `register_hanging_slot=False` 时 `len(hanging_ids) == 0` | `BaseModel.build_inputs` |
| 24 | `fim_layers >= 1` | `__init__` |

---

## 9. 刻意的设计选择与陷阱

### 9.1 为什么这样设计

| 选择 | 理由 |
|---|---|
| user/item token 全程保持独立 batch 轴 | serving 时 user 侧只需算一次（R=1），避免 ×C 的重复计算与显存 |
| `user_global` 640 维切 5 chunk | chunk 0 做 user token，chunk 1-4 与 item cache 的 chunk 1-4 配对生成 4 个 fusion query，与 item cache 共用同一套切分口径 |
| item cache 训推分离 | serving 不必重算 60 个 item 特征（816 维），只查 640 维表 |
| `atc` 保留 native 50 + 额外池化出 25 | native 50 作 dedicated route给 fusion query `atc`；25 进 generic memory 控制总长度 |
| dedicated / generic 双视图 | 4 个 fusion query 各只看自己那条行为序列（强相关），4 个 base token 看全部 200 token（全局） |
| UIHeadMixer 无参数 | 通道置换用常量索引 gather，不引入参数也不引入 launch |
| `user_tail` 在 M-FFN 之前转移 | 把 COMMON→ITEM 的对齐点压到唯一一个加法，且只传 64 维（不是完整 128） |
| TIM 7 query 里含 dd/pdp/pp | 场景专用表征，经硬路由选一个；`common` 进 gate_input |
| 任务塔双层门控 `2*sigmoid` | 值域 (0,2)，既能抑制也能放大，比标准 sigmoid 门控表达力强 |

### 9.2 已知陷阱（重写时极易踩）

1. **序列组里的静态特征会被 EGO 静默丢弃** —— 所以 `build_group_input` 必须 raise。同理，长度不齐的 concat 会静默错位。
2. **`ego.Normalization` / `INormalization` 不能用在四类张量上**：① 序列组（moments 在 batch×time 上统计，padding 污染统计量）；② 冻结的多模态 embedding（下游要 `l2_normalize` 算 cosine，而 cosine 对减均值敏感）；③ assign slot / item cache（值由 teacher 每步写入，分布非平稳，moving stats 永远滞后）；④ 用于硬路由的 one-hot（减均值后 0/1 语义丢失）。
   - 本模型当前对 `brv4_scene_token` 与 `click_mean`/`order_mean` 加了 `input_norm=True`，属于①④ 的边界情形；硬路由本身读的是原始 `scene_onehot_input` 所以没坏，但这是**已知的语义妥协**，不是最佳实践。
3. **`tf.cast(float→int32)` 是向零截断，不是 floor**。负值会截到 0。SimTier 的索引计算依赖这一点。
4. **SimTier 的 cosine 不能用 `matmul` 算**。硬分箱是离散映射，~1e-7 的 fp 差异会翻转桶归属。
5. **`unsorted_segment_sum` 不能进 serving 图**（tf2onnx 经 Unique 下降，TRT 无法解析）；`ScatterNd` 也不行（缺 additive reduction）。所以必须训推分实现。
6. **rank-3 / rank-4 可训练变量 + batched matmul 会触发 EGO XLA CPU sandbox 的 `TransformMmt4D` 崩溃或静默失效**。所以：
   - `tim_block` 全部用 `_independent_kernels`（rank-2 变量），只在 **serving** 路径用 `tf.stack` 拼成 rank-3（非变量）
   - `fim_block` 用 `add_weight` 的 rank-3 变量但**先切片到 rank-2** 再进 matmul（`route_kv_kernel[route_idx]`），或用 `grouped_linear` 的 batched 形式
   - 两者都是已验证可行的模式，不要自创新组合
7. **`tf.concat` 不会被 CSE 合并如果 `name` 不同**。三个 order 类任务的 `tower_input` 内容相同但名字不同，所以算了三次（已知冗余）。
8. **`tf.zeros_like` 的粒度陷阱**：对齐锚点必须用与目标同 batch 的**小张量切片**（如 `item_kv[:, :1, :]`）。`simtier_v3` 的注释警告过：用 `zeros_like(q[:,None,:])` 会反而把 K 扩到 item batch。
9. **`type_embedding * (type_index+1)` 不省参数**。它在 per-group scope 内创建，5 条流各有独立变量；乘子只造成初始幅值 1~5 倍的不均衡。
10. **`_pool_fixed` 的 mask 返回 float32，不是 bool**。下游契约依赖这一点。

---

## 10. 重写验收清单

按顺序执行，每一步都能独立失败定位：

### 10.1 建图期

- [ ] `python3 model_v1m4.py` 能跑到 `print_total_params()` 不报错
- [ ] 可训练参数总量 = **11,437,381**（训练）/ **10,914,501**（serving），非训练 moving stats = **20,744**
- [ ] 全部 24 条不变量断言（§8）都存在且能触发（逐个造错验证）
- [ ] `configs/export.yaml` 的 `sparse.slot_ids` 含 30102，`dense` 为 `[{export_name: scene_onehot, slot_ids: [53903]}]`

### 10.2 形状契约

- [ ] `user_tokens` = `[B_u, 8, 128]`，`item_tokens` = `[B_i, 8, 128]`
- [ ] `sequence_tokens` 的 6 个视图：`click[B,50,128]` `order[B,50,128]` `long_click[B,50,128]` `atc[B,50,128]` `atc_generic[B,25,128]` `upstream[B,25,128]`，对应 mask `[B,·]` 且 **dtype=float32**
- [ ] FIM 输出：`user_final [B_u,8,128]`、`item_final [B_i,8,128]`、`monitor {}`
- [ ] `rank_global` = `[B_i, 256]`
- [ ] `tim_tokens` = `[B_i, 7, 128]`
- [ ] `scene_condition [B,3]`、`scene_hidden [B,128]`、`gate_input [B,259]`
- [ ] `tower_inputs` 宽度 = `[640, 768, 896, 896, 896]`
- [ ] `grouped_predictions` = `[B,5,1]`，5 个 `p0_{target}` 各是 `[B,1]`
- [ ] SimTier：`img_score [B,231]`、`title_score [B,231]`、`long_image_score [B,204]`

### 10.3 变量名契约（checkpoint 兼容）

- [ ] 对比 `tf.compat.v1.global_variables()` 的名字集合与参考实现完全一致
- [ ] 重点核对三处手写 `get_variable` 与历史 `DenseTower`/`tf.layers.dense` 的命名对齐：
  - `brv4_fusion_queries/{route}_query/{kernel,bias}`
  - `brv4_rank_global_mlp/layer_{1,2}/{kernel,bias}`
  - `p0_{target}_brv4_tower/{hidden_0,gate_0_hidden,gate_0,hidden_1,gate_1_hidden,gate_1,output}/{kernel,bias}`
- [ ] `DenseTower` 产生的名字是 `{name}/layer_{i}/{kernel,bias}`，`norms=[True,...]` 时额外有 `{name}/input_norm/*`

### 10.4 数值契约

- [ ] `python3 tests/tim_numpy_contracts.py` → 15/15 OK（纯 NumPy，无需 EGO）
- [ ] `python3 -m unittest tests.tim_path_contracts -v` → 7/7 OK（需 EGO 镜像）
- [ ] `python3 -m unittest tests.tim_qk_fold_contracts -v` → 9/9 OK（需 EGO 镜像）
- [ ] `python3 -m unittest tests.tf_fim_contracts -v` → 4/4 OK
- [ ] `python3 tests/tf_fim_contracts.py --xla` → CPU XLA 编译的前向 + 输入/参数梯度全部有限
- [ ] SimTier 特征值对齐：新旧实现在同一样本上的 231/231/204 维输出**逐元素相等**（整数计数派生，不是近似）
- [ ] 池化对齐：`_pool_fixed` 与逐桶循环版在 `L ∈ {0,1,2,3,7,9,49,50,src/2,src-1,src}` 全场景下 value 与 pooled_mask 逐比特相同

### 10.5 serving 契约

- [ ] `tests/compiled_fim_checks.py::check_memory` 对真实 `graph.pb` 断言无 `Tile`，且 `len(memories) == 8`（2 层 × 每层 generic+dedicated 两对）
- [ ] ⚠️ **当前该检查只覆盖 FIM 的 `uniformer_request_attention` scope**。要验证 TIM 与 SimTier 的请求级收益，需把过滤条件扩到 `brv4_request_tim_attention` / `brv4_short_cosines` / `brv4_long_cosines`，并把 SimTier 的叶名 `memory` 纳入集合
- [ ] 多请求隔离：两请求各三候选，修改一个请求的输入不能影响另一个请求的输出
- [ ] 候选数不同的请求：`reshape` 应报错而不是静默错配

### 10.6 端到端

- [ ] 训练能跑起来且 loss 下降；5 个任务的 GAUC 均 > 0.5
- [ ] 梯度全部有限（无 NaN/Inf）
- [ ] `item_cache_norm` 作为预测 target 能导出，且量级合理（反映 cache 写入是否正常）
- [ ] serving 导出后 TRT 能成功构建（重点看 SimTier 的分块 Equal 路径与 `Gather` 算子）

---

## 附：相关文档

| 文档 | 读者 | 内容 |
|---|---|---|
| **`model_intro.md`** | **人** | 架构与设计思想：四条主线、五个关键决策的"为什么"、训推两套图、规模画像、思想谱系、已知妥协。配 ASCII 示意图。**先读这个** |
| `model.md`（本文） | AI / 重写者 | 实现规格：逐行 shape、变量名、初始化器、算子顺序、24 条不变量、验收清单 |
| `new_optimize.md` | AI / 审查者 | TIM 两条 attention 路径的等价性证明（代数 + 240 用例数值验证）、QK 结合律折叠方案与 MAC 核算 |
| `optimize.md` | AI / 审查者 | 从 uniformer 借鉴的效率优化待办（A–H，G 已排除） |
| `FIM.md` / `TIM.md` | — | FIM 与 TIM 的历史设计文档（部分内容已过期，以本文与源码为准） |
| `plan_v1m4_brnew_v4.md` | — | v1m4 的原始设计规划（历史文档） |
| `tests/` | CI / 审查者 | 5 个契约测试 + `param_inventory.py`，见 §10.4 |

本文档**自包含**：不依赖任何仓库外文件，所有 mcconf 组维度与参数量均已硬编码进正文，重写时无需再读 yaml 或运行清点脚本。

本文档对应的源码状态：`rankmixer_tim_model_v1m4.py` 899 行、`fim_block.py` 390 行、`tim_block.py` 290 行、`request_attention.py` 163 行、`simtier_v1m4.py` 192 行。
