# Whisper 跨平台 benchmark harness

配套报告：[`docs/perf/whisper-cross-device-20260827.md`](../../../docs/perf/whisper-cross-device-20260827.md)

在 Hailo-8 / RK3588 / RK3576 / Jetson / 树莓派 CPU 上跑同一批语料、同一套评分的 Whisper 实测工具。

## 设计约束

**设备只产出转录文本和分段计时，打分统一在一台机器上做。** 这是刻意的：让每台设备各装一套 jiwer/cn2an/opencc 会导致口径漂移，而口径漂移在跨设备对比里是致命的。设备侧的依赖因此压到最低——numpy + 各平台自己的 runtime，不装 torch、不装 librosa。

语料是 `bench/perf/corpus`（sha256 钉死），评分逐行复刻 `bench/perf/runners.py`。

## 文件

| 文件 | 用途 | 设备侧依赖 |
|---|---|---|
| `rknn_whisper_run.py` | RK3588 / RK3576，RKNN encoder + RKNN 或 CPU ONNX decoder | numpy、rknnlite、onnxruntime（hybrid 档） |
| `hailo_corpus_bench.py` | Hailo-8，HEF encoder + CPU ONNX decoder，含 VAD 分段 | numpy、hailo_platform、onnxruntime、librosa |
| `wcpp_corpus_run.py` | Jetson，包装 whisper.cpp CUDA，解析其分阶段计时 | 仅 python 标准库 + numpy |
| `trt_whisper_run.py` | Jetson 裸 TensorRT 三引擎管线 ⚠️ **见下方状态** | tensorrt、cuda-python、numpy |
| `score_all.py` | 统一打分，输出 CER/WER + RTF + TTFT | jiwer、cn2an、opencc |

## 用法

```bash
# RK：--encoder_duration 必须与 .rknn 编译时的窗口一致，不是自由参数
python3 rknn_whisper_run.py --corpus corpus --lang en \
  --encoder model/whisper_encoder_base_20s.rknn --decoder onnx_dec \
  --vocab-dir model --encoder_duration 20 --all-cores \
  --label rk3588-hybrid-en --out results/hybrid_en.json

# Jetson whisper.cpp：不要传 -np，它会关掉 whisper_print_timings
python3 wcpp_corpus_run.py --corpus corpus --lang en \
  --bin ./whisper.cpp/build/bin/whisper-cli --model models/ggml-base.bin \
  --label orin-nano-wcpp-base-en --out results/wcpp_base_en.json

# 打分
python3 score_all.py 'results/*.json'
```

`--decoder` 传 `.rknn` 文件走全 NPU，传目录则走 CPU 上的 optimum ONNX（KV cache）——后者在实测里两个维度都更优。

## 默认开启的三个防护

内容不足的音频会让 Whisper 不吐 EOS，进而复读或胡编。这个失败模式与硬件厂商无关，在补零（Hailo 短句填满固定窗口）和截断（RK 10s 切块后的尾块）两条路径上都会触发，所以做成默认而非平台补丁：

1. **按时长定 token 预算** `min(cap, 时长×8+12)`。固定上限只防崩溃不防失控——Whisper 位置表只有 448 项，超出会直接报 `idx=448 out of data bounds`。
2. **相似度复读清理**。Hailo 官方的 `clean_transcription` 只判子串包含，抓不到自我改述；且只切 `.` `?`，中文无句读直接漏过。这里用 difflib 相似度 + CJK 句读。
3. **句内循环守卫**。按 n-gram 检测重复 ≥3 次的短语并截断——句级去重看不见 `by Llew, by Llew, ...` 这种句内循环。

## `trt_whisper_run.py` 注意事项

**whisper-base 的 encoder 引擎要用 `--bf16` 构建，且不能同时给 `--fp16`。**

`trtexec --fp16` 从 base 的 encoder ONNX 建出的引擎，输出与 onnxruntime 的 cosine 只有 **0.826**，且 run-to-run 确定——是 fp16 kernel 选型的精度问题（fp16 只有 5 位指数）而非竞态。`--bf16` 把指数位拿回到 8 位，cosine 回到 **0.9996**，端到端 err 与 fp32 **逐位相同**，而 encoder 只从 10.75 ms 变成 12.53 ms（fp32 是 39.1 ms）。

**同时给 `--fp16 --bf16` 无效**：TRT 全程选 fp16，产出的引擎与纯 fp16 逐位相同（cosine/maxabsdiff/std 三个值一模一样）。这与直觉相反，别指望 TRT 会自己挑需要大指数范围的层。

**这个缺陷极具欺骗性**：坏 encoder 仍输出一个看起来正常的张量，decoder 照常解出语法通顺的英文、只是内容漂移并提前 EOT，肉眼与「KV cache 没累积」无法区分。tiny 的 30s/10s fp16 引擎和两个 decoder 引擎都正常（cosine 0.9999），所以**是模型特异的**。

**把「对着 onnxruntime 做数值对拍」作为 TRT 引擎构建流程里的常设一步**，不要凭精度标志推断。

```bash
# base encoder：bf16，不要加 --fp16
trtexec --onnx=enc_base_30s.onnx --bf16 --shapes=input_features:1x80x3000 \
        --saveEngine=enc_base_30s_bf16.plan
# tiny encoder：fp16 可用
trtexec --onnx=enc_tiny_30s.onnx --fp16 --shapes=input_features:1x80x3000 \
        --saveEngine=enc_tiny_30s.plan
```
