# Turn-Driver Unification — server-loop / client-loop 单一实现

状态: DESIGN (待 codex review → 实施)
作者: 主线程 (CTO) 2026-06-14
关联: `docs/plans/conversation-split.md`(voxedge 引擎拆分谱系)

## 1. 问题

LLM↔tool 多轮 pump 当前有**两份手写实现**,算法 ~90% 重合且已在漂移:

| | 文件 | 形态 | 宿主 |
|---|---|---|---|
| client pump | `agent/ovs_agent/tools/runner.py:90` `run_tool_loop()` | free function + 回调 seam | `app_mode.py:148-420` 本地跑 |
| server pump | `voxedge/engine/llm_turn.py` `_LLMTurn.run()` | back-ref 紧耦合 `Session` | `ConversationEngine` 内 |

voxedge `_LLMTurn` 的 docstring(llm_turn.py:4-6, 58-62)自承"Ported in shape from
`agent/openvoicestream_agent/tools/runner.py:116-443`"。血缘明确,是同一算法的两份拷贝。

**历史代价**:memory 记录的衰退 bug(preamble advertise 剥字段、barge-in 协作取消、
tool_result 错投、nothinking 双发)几乎全长在"一边改另一边没同步"这条缝上。

## 2. 共同算法(两份都有,逐项对齐)

1. `_ToolCallAcc` 累积器 —— runner.py:42 / llm_turn.py:28,**字段完全相同**
2. `stream_events(messages, tools=schema)` 事件流,kind ∈ {text, tool_call_delta, finish} —— 接口相同
3. 文本 chunk → 文本 sink
4. tool_call_delta 按 index 累积
5. preamble 早发(拿到 tool name 即发)+ dispatch 兜底 —— runner.py:223 / llm_turn.py:142,206
6. `registry.dispatch(name, args, ctx)` —— 签名相同
7. template 快路:全部工具 `response_mode="template"` + 非空 `completion_text` + 成功 → 跳 round2 —— runner.py:288 / llm_turn.py:254
8. max_iterations / max_tool_rounds 上限
9. barge / cancel 中止

## 2b. 已发生的行为漂移(codex review 揪出 —— 不是"同一算法两拷贝")

两份 pump **今天就已经行为不一致**,统一时必须显式选定一套语义,不能当机械合并:

1. **preamble 去重维度不同**:client 按 tool_call **index** 去重(runner.py:133,228),
   server 按 tool **name** 去重(llm_turn.py:100,142)。同名两次调用,两边行为不同。
2. **template 快路语义不同**:server 要求**全部**工具 template 且把所有 completion **join**
   后说出(llm_turn.py:247);client 是**任一** template 即走、且**取首条**(runner.py:376,429)。
3. **system-prompt 前置**是 driver 可见规则,不只是 MessageSink 管线(llm_turn.py:81,app_mode.py:332)。

→ **P0 抽取必须保持 server 语义(name 去重 / all+join)字节等价;P1 与 client 的语义差异是
真实行为决策,需单列评审,不能默默合并。**

## 3. 真正的差异(= 要抽象的 seam)

| seam | client (runner.py) | server (llm_turn.py) | 统一方案 |
|---|---|---|---|
| LLM 后端 | 任意 backend(EdgeLLM/OpenAI/Noop) | `sess._llm_be`(EdgeLLM 写死) | 注入 `llm`(已都是 `.stream_events`) |
| 文本 sink | `slv.send_text`(回灌 SLV TTS) | `sess._tts.enqueue_text`(引擎 TTS buffer) | `text_sink: async (str)->None` 回调 |
| preamble sink | `on_tool_preamble` 回调(已是 seam) | `self._emit_preamble→sess._tts` | 复用 client 的回调形态 |
| completion sink | `on_tool_completion_text` 回调 | `sess._tts.enqueue_text` | 同上 |
| message-state | `session.add_assistant_tool_calls()` + cancel 回滚 | 本地 `messages` list,append-only,barge 即 return 丢弃 | `MessageSink` 协议(见 §4) |
| **barge 信号** | `asyncio.Task.cancel()` → CancelledError → 回滚 | 协作 flag `sess.state.llm_barged` 轮询 | **统一为协作 `should_abort()` 回调**(见 §5) |
| tool registry | agent `default_registry`(本地 dispatch) | voxedge `ToolRegistry`(local + remote dispatch) | 注入 `registry`,dispatch 语义已对齐 |

## 4. 目标架构

**结论(承接上轮架构判定):turn driver 归 voxedge。一份实现,两个接线点。**

```
voxedge/engine/turn_driver.py   ← 新增:provider-agnostic、无 I/O 的纯 pump
   run_turn(llm, registry, messages, *, sinks, msg_sink, should_abort, cfg)
        │
        ├── voxedge/engine/llm_turn.py  ← _LLMTurn 退化为薄 adapter:
        │       把 sess._tts / sess.state.llm_barged / sess.transport 接到 seam
        │
        └── agent/ovs_agent/app_mode.py ← client-loop 调同一 run_turn:
                text_sink=slv.send_text, msg_sink=agent Session,
                should_abort=<协作 flag>, registry=default_registry
                → 删除 agent/ovs_agent/tools/runner.py 的 pump(或留薄 shim)
```

seam 协议(driver 只认这些,不认 Session/engine):

```python
class TextSink(Protocol):
    async def text(self, s: str) -> None: ...        # assistant 文本 + completion_text
    async def preamble(self, s: str) -> None: ...    # 工具 preamble
    async def flush(self) -> None: ...               # 回合末 flush/signal

class MessageSink(Protocol):
    def add_assistant_tool_calls(self, content, tool_calls) -> None: ...
    def add_assistant_text(self, content) -> None: ...
    def add_tool_result(self, tool_call_id, content) -> None: ...
    def working_messages(self) -> list[dict]: ...    # 发给 LLM 的当前 list

# should_abort: Callable[[], bool]  —— 每个 await 点前轮询
```

**seam 契约补强(codex must-fix #1)**:driver 入参须显式带两个**策略参数**,不能写死:
- `preamble_dedup: "index" | "name"`(P0 server 传 "name")
- `template_fastpath: "all_join" | "any_first"`(P0 server 传 "all_join")
这样 P0 保 server 字节等价,P1 切 client 时语义差异变成一个**显式可评审的参数翻转**,而非隐性漂移。

**system-prompt 归属(codex must-fix #2 / D3)**:system-prompt 前置是 **caller 的责任,
不在 MessageSink 内**。caller 在调 `run_turn` 前就把 system 放进 `working_messages`
首位(client: `session.messages(system_prompt)` app_mode.py:332;server: llm_turn.py:81-83),
driver 只 append assistant/tool,绝不自己插 system。MessageSink 协议**不含** add_system。

## 5. 关键设计决策(请 codex 重点审)

**D1 — barge 机制(codex:P0/P1 不要切协作式)。**
client 靠 `task.cancel()`→`CancelledError`→回滚 session+working messages(runner.py:473);
app_mode 仅在 cancel/error 路径 abort SLV 缓冲文本(app_mode.py:359,388)。
协作轮询**无法及时打断** client 的 LLM `__anext__()` 和 tool dispatch await(runner.py:171,332),
且协作 return 在 append 了 assistant/tool 之后会留脏状态(runner.py:277,337)。
**结论(采纳 codex)**:P0/P1 **保留两侧各自的 barge 机制**——`should_abort()` seam 在
server 侧接 `state.llm_barged` 轮询,在 client 侧由 driver 在中止点抛/传播 cancel 让 caller 回滚。
driver 不强制统一 barge 语义。协作化收口推迟到 P2,且须先解决 client in-flight await 的及时中止。

**D2 — agent 是否接受 voxedge 依赖?**(load-bearing)
统一到 voxedge 意味着 `ovs_agent` client-loop 要 `import voxedge.engine.turn_driver`。
- 若接受 → 一份 driver,两处接线,衰退面减半(推荐)。
- 若不接受 → driver 须放第三方共享小包,两边都依赖(依赖图更碎,不推荐)。
- 备选 → 判定 client-loop 已无生产/CI 真实调用方,直接删 client pump,voxedge 维持唯一实现(最省,但需数据支撑)。
**codex 倾向**:P0/P1 **保留 client pump**,voxedge 设为硬依赖**只在 D1/D3 完全定稿后**再做。
BYO-LLM 拓扑是真需求——app_mode 解析任意 `self.llm`、本地 registry、allowlist、本地 `ToolCallCtx`
(app_mode.py:213,227,337),server-loop 是显式 opt-in(app_mode.py:95,113)。
→ **P0 完全不碰 client,纯 server 侧抽取;client 接线(P1)留到 D2 用户拍板后。**

**D3 — message-state 所有权差异。**
server append-only(server 拥有对话状态);client 写进 agent `Session.history`。
`MessageSink` 协议要同时容纳两者,且 system-prompt 前置逻辑(llm_turn.py:81-83)只能发生一次。

**D4 — 行为零变化验收。**
driver 抽取必须逐项 byte-for-byte 等价。两侧各有回归:
- server: `tests/` v2v server-loop 场景 + bench
- client: `test_server_loop_integration` / runner 单测 + 真机多轮

## 6. 实施分期(每期可独立 ship + 回归)

- **P0**:抽 `turn_driver.py`(纯算法 + seam 协议),voxedge `_LLMTurn` 改为薄 adapter。
  仅 server 侧改动,client 不动。server 回归绿 = P0 通过。**零行为变化。**
- **P1**:client-loop 的 `app_mode.py` 切到 `run_turn`,删 `runner.py` pump。
  依赖 D2 拍板。client 回归 + 真机多轮绿 = P1 通过。
- **P2**(可选):D1 barge 协作化收口、preamble 低延迟接口(llm_turn.py:48 TODO)。

P0 风险最低、先行止血;P1 才触及跨仓依赖决策。

## 6b. P1 实施设计(codex 设计 + 主线程决断,已定稿)

**目标**:client-loop(`app_mode.py`)改走 `voxedge.engine.turn_driver.run_turn`,删 agent 自写 pump
(`runner.py:88 stream_with_tools`)。agent 接受 voxedge 依赖(裸装=numpy-only)。

**seam 超集扩展(Q1)**:`run_turn` 末尾新增 5 个可选参数,**全默认 None → server 路径零感知、P0 字节等价**:
`on_tool_started`, `on_tool_completed`, `first_token_timeout_s`, `idle_timeout_s`, `on_timeout`。
- 把 `turn_driver.py` 的 `async for ev in llm.stream_events(...)` 改为显式 `__aiter__()`,
  首 token 前用 `asyncio.wait_for(__anext__(), first_token_timeout_s)` 包,首 payload 后每次 `__anext__()` 施 idle 超时(对照 runner.py:154-184)。超时触发 `aclose()` + `on_timeout(...)`。
- `on_tool_started(tc)` 插在 preamble/json.loads 之前(对照 runner.py:292);
  `on_tool_completed(tc,result,dt_ms)` 插在 `add_tool_result` 之后(对照 runner.py:335)。

**cancel/回滚(Q2/D1)→ 决断:Option A**。driver 透传 `CancelledError`(py3.10 它继承 BaseException,
现有 `except Exception` 不吞,已验证);**回滚由 app_mode caller 在 `run_turn` 外快照+恢复 session**,
不进 MessageSink(保持 provider-agnostic)。`app_mode.py:359-387` 的 `slv.abort()`/`stop_playback`
是音频清理、不是历史回滚,**保持在 caller 的 `except CancelledError` 分支**。

**错误透传(主线程决断,解 codex 风险①)→ 新增 `reraise_errors: bool = False`**:
server adapter 传 False(维持 P0 的 swallow+flush 行为,字节等价);client 传 True
(driver re-raise → app_mode.py:388 的 `except Exception` 接管做 slv.abort)。turn_driver 唯一消费者
是 `_LLMTurn`,此参数只影响 server/client 两路,不波及他处。

**MessageSink 适配 Session(Q3)**:agent 侧 MessageSink 包 `messages_for_llm = session.messages(system_prompt)`
(app_mode.py:332 已前置 system,D3:caller 责任),`working_messages()` 返回它;
`add_assistant_tool_calls`/`add_tool_result`/`add_assistant_text` 同时 append working list + 写 `session.*`
(对照 runner.py:255,277,337)。

**策略参数修复(主线程决断,解 codex 风险②)→ 只改 client 路径,server(name/all_join)不碰**:
- `preamble_dedup="index"` fallback 必须用**原始 tool_call index**(构造 tc_payload 时保留原始 idx),
  不能用 `enumerate(tc_payload)` 位置——稀疏 index 不等价(对照 runner.py:133,228,287)。
- `template_fastpath="any_first"` 补 runner.py:383 的护栏:模板工具失败/空 completion 则放弃 fast-path。
这两处只在 index/any_first 分支,**server 的 name/all_join 路径字节不变**。

**落地步骤**:① `agent/pyproject.toml` 加 `voxedge` dep;② `turn_driver.py` 扩 5 seam + `reraise_errors`
+ 修 index fallback + 补 any_first 护栏;③ `app_mode.py` 实现 agent MessageSink + caller 快照/恢复 +
改调 `run_turn`;④ `runner.py` 先留 shim、测试全绿后删 pump。

**回归门**:agent `test_session_tools` / `test_barge_during_tool` / `test_llm_stream_events` /
`test_server_loop_*` 全绿 + voxedge P0 测试(49 targeted)仍绿。最敏感:barge/cancel 回滚 + timeout 流迭代。

## 6c. P1 STOP 裁决(执行体发现 client 有 5 个 agent 私有概念)

执行体实施前发现:client `stream_with_tools` 携带 server **结构上没有**的 agent 私有概念,
且真正 pin 其契约的是 `agent/tests/test_tools_runner.py`(20 tests,§6b 漏列)。裁决如下。

**路线:完全合并(用户拍板)+ 行为保持。** 循环骨架抽进 `run_turn` 共享;两侧差异用策略参数表达,
**两边各自字节等价,零行为变化**。第一类私有概念留在 agent shim,绝不污染 driver。

**5 个 agent 私有概念 → 做成 run_turn 的通用可选 seam(server 全传 None/默认 → P0 不变)**:
- B1 allowlist → **不进 driver**。改为 caller 预解析:`run_turn(tools_schema=...)` 可选;
  server 传 None → driver fallback `registry.list_openai_tools()`(P0 行为);client 传 allowlist 过滤后的 schema。
- B2 prefix-cache `extra_body` → 通用 seam `llm_params_for_round: Callable[[int], dict] | None`。
  driver 每轮 merge 它返回的 dict 进 llm_params,**driver 不知道 prefix-cache 是什么**;server 传 None;
  client 传一个 round>0 注入 `save_system_prompt_kv_cache` 的 fn。
- B3 iteration-limit event → 通用 seam `on_iteration_limit: Callable | None`(server None=仅 flush=P0;
  client 发 EventBus 事件)。**上限回滚由 caller 做**(同 cancel 的 Option A)。
- B4 session kwarg → 经 llm_params 透传即可(client 把 session 放进自己的 llm_params),非问题。
- B5 返回 final text → `run_turn` 返回 `Optional[str]`;**server adapter 忽略返回值**(纯附加,
  不改 server 副作用 → P0 字节等价);client 用它满足 test_tools_runner 的 `final == ...` 断言。

**`stream_with_tools` 不删,退化为薄 shim**:保住 test_tools_runner.py 20 个测试的契约
(allowlist→schema、prefix-cache fn、iteration event、session、返回值都由 shim 提供),
内部循环骨架 delegate 给 `run_turn`。→ **test_tools_runner.py 保持全绿,不删测试。**
"删 pump"实际是"把 ~150 行循环骨架换成 run_turn,shim 只剩 ~agent 私有适配"。

**行为对齐(第二类 §2b)= 独立后续决策,不进本次重构**:preamble dedup(index vs name)、
template fast-path(any_first vs all_join)两处真实行为差异,本次**只用参数表达、各侧不变**。
是否收成一种(真对齐)单列 UX 决策评审。**严禁在 refactor 里悄悄改某侧行为。**

**回归门修正**:真正的 client pump 门是 **`agent/tests/test_tools_runner.py`(20)**,
加上 `test_session_tools`/`test_barge_during_tool`/`test_llm_stream_events`/`test_server_loop_*`,
以及 voxedge P0 的 49。baseline:agent gate 87 passed、test_tools_runner 20 passed、voxedge P0 49 passed。

**prefix-cache 更正(原误判已纠)**:server-loop **已有** prefix-cache,在 backend 层无条件设
(`server/core/edge_llm_backend.py:138` `body.setdefault("save_system_prompt_kv_cache", True)`,
docstring:Matches agent edge_llm.py cold path)。client 是 backend(iter0)+ pump(`iter>0`,
runner.py:149-152)。**净行为相同:两侧每轮都存 prefix KV。** 不存在"server 白慢"的 gap,该后续项作废。
合并经 `llm_params_for_round` seam 各保留自己的层:client 传 iter>0 注入器,server 传 None(backend 已管)。

**教训**:声明"某侧缺 X"前必须 grep 原代码确认无等价物——B2 差点据此引出一个伪 server 提速项。
其余私有概念也已逐一核对:B1 allowlist(server 用同一 `list_openai_tools` 只是传 None 不筛,非缺失,
浅层差异)、B3 iteration event(server 有 warning 日志 + append-only 故无回滚,layer/ownership 差异,合法)、
B5 返回值(server 返回 None 走 TTS,附加不冲突)。

**6 个私有概念逐条对 server 原代码核实(用户要求,不许假设)**:
| 概念 | server 实际 | 真无? |
|---|---|---|
| B1 allowlist | 同一 `list_openai_tools`,不传 allow 不筛(tool_registry.py:259-266) | 非缺失,同方法不同参 |
| B2 prefix-cache | backend 无条件设(edge_llm_backend.py:138) | **server 有**(曾误判) |
| B3 iter event+回滚 | 仅 warning 日志、append-only 无回滚无事件 | ✓ 真无(所有权差异) |
| B4 session kwarg | backend 签名 `session unused server-side`(edge_llm_backend.py:153) | ✓ 显式不用 |
| tool 事件 | conversation/client_events 不 emit | ✓ 真无 |
| 超时 first/idle | **httpx 60s read 超时**(edge_llm_backend.py:82,107),非语义级 | **server 有粗的**(曾会误判"无") |

## 6d. P2 行为对齐(用户已拍板,合并落地后独立 commit,**不进零变化的 P1**)

三处行为差异,方向已定。P2 是独立的 behavior-change commit,每处带测试更新,**不与 P1 混提**
(P1 必须保持两侧字节等价以隔离验证)。

1. **preamble 去重 → 对齐 server(name)**:client 从 index 翻到 name。
   影响:同一轮同名工具重复调用只发一遍 preamble(不再两遍)。仅边缘场景,常见单工具路径不变。
2. **template 快路 → 对齐 server(all_join)**:client 从 any_first 翻到 all_join。
   影响:混合 template+非 template 一轮不再被 template 短路吞掉非 template 结果;改跑 round-2 综合。
3. **超时 → 对齐 client(语义级)**:server-loop adapter 把 timeout seam 从 None 改为
   15s first-token / 30s idle + on_timeout(surface 优雅错误)。合并后 server adapter 一行翻参数即可。

**P2 副产物(简化)**:§2b 两处统一到 server 语义后,`preamble_dedup` / `template_fastpath` 策略参数
+ `index` / `any_first` 分支全部变成 dead code → **P2 一并删除**,driver 回到单一语义,divergence 面归零。
(P1 仍保留这些参数/分支以维持 client 当前行为字节等价;P2 才翻参数 + 删分支。)

## 7. 待确认(codex review 输出项)

1. D1/D2/D3 的倾向与风险点,带 file:line。
2. seam 协议是否覆盖两侧所有耦合点(漏一个就是下一个衰退源)。
3. P0 抽取的 byte-equivalence 边界:哪些 await 顺序/flush 时机不能动。
4. agent `Session` 在"协作中止 + 不 cancel"下的状态正确性(D1 风险)。
