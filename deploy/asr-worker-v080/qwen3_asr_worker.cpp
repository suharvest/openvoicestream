/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_vad_split.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "mel_extractor.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "requestFileParser.h"
// v0.8.0 migration: the v0.7.1-era streaming spec-decode runtime and slot pool
// were removed from the engine baseline. This serving worker now drives the
// v0.8.0 vanilla rt::LLMInferenceRuntime on the ONE-SHOT path only (production
// runs stream_mode=accumulate → the one-shot `requests` path is the live one;
// the streaming begin/chunk/end protocol is dormant and returns a 501 stub).
#include "runtime/asrStreamingSessionRuntime.h" // SessionLaneManager (lane reservation only)
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <optional>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{
//! Global MelExtractor handle. Loaded once at startup if the caller passed
//! --melSettings / --melFilters (or set the matching env vars). Null means PCM
//! input is disabled and `pcm_b64` chunks are refused with pcm_input_unsupported.
//! Restored from the v0.7.x streaming worker (asr-worker-build-verify): the #9
//! refactor dropped this, so `pcm_b64` chunks fell through to the request parser
//! as a non-safetensors `audio` value and SIGABRT'd. The production
//! StreamingWorkerSession (voxedge trt_edge_llm_asr.py) sends raw float32-LE PCM
//! via `pcm_b64`, so the worker MUST convert PCM->mel itself in stream_mode=worker.
std::unique_ptr<MelExtractor> gMelExtractor;

struct Args
{
    std::string engineDir;
    std::string multimodalEngineDir;
    std::string melSettingsPath;     //!< whisper_feature_extractor.json (optional, enables PCM input)
    std::string melFiltersPath;      //!< mel_filters.bin (optional, enables PCM input)
    int32_t maxSlots{4};             //!< ASR decoder slot-pool size (D1 concurrency). Default 4.
    bool debug{false};
};

//! Minimal base64 decoder (RFC 4648; ignores '=' padding and whitespace).
//! Restored verbatim from the v0.7.x reference worker — used to decode the
//! `pcm_b64` payload (raw float32 little-endian PCM) the service sends.
std::vector<uint8_t> base64Decode(std::string const& in)
{
    static int8_t kT[256];
    static bool kInit = []() {
        for (auto& v : kT) v = -1;
        char const* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) kT[static_cast<unsigned char>(a[i])] = static_cast<int8_t>(i);
        return true;
    }();
    (void)kInit;
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4 + 4);
    int32_t v = 0;
    int32_t bits = 0;
    for (char c : in)
    {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        int8_t const x = kT[static_cast<unsigned char>(c)];
        if (x < 0) continue;
        v = (v << 6) | x;
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((v >> bits) & 0xFF));
        }
    }
    return out;
}

//! Write a [1, n_mels, n_frames] fp16 mel as a single-tensor safetensors file
//! named "mel" — the exact on-disk format the requestFileParser audio path
//! consumes (.safetensors mel-spectrogram). Restored verbatim from the v0.7.x
//! reference worker so the PCM-derived mel is byte-compatible with the mel_path
//! fixtures the standalone gates used.
void writeMelSafetensors(std::vector<float> const& mel_f32, int32_t n_mels, int32_t n_frames,
    std::filesystem::path const& out_path)
{
    auto f32_to_f16 = [](float f) -> uint16_t {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        uint32_t const sign = (x >> 16) & 0x8000u;
        int32_t const exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
        uint32_t const mant = x & 0x7FFFFFu;
        if (exp <= 0)
        {
            if (exp < -10) return static_cast<uint16_t>(sign);
            uint32_t const m = (mant | 0x800000u) >> (1 - exp);
            uint32_t const rounded = (m + 0x1000u) >> 13;
            return static_cast<uint16_t>(sign | rounded);
        }
        if (exp >= 31)
        {
            if (((x >> 23) & 0xFF) == 0xFF && mant != 0) return static_cast<uint16_t>(sign | 0x7E00u);
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        uint32_t const m = mant >> 13;
        uint32_t const r = mant & 0x1FFFu;
        uint16_t out = static_cast<uint16_t>(sign | (exp << 10) | m);
        if (r > 0x1000u || (r == 0x1000u && (m & 1))) ++out;
        return out;
    };

    int32_t const batch = 1;
    size_t const elem_count = static_cast<size_t>(batch) * n_mels * n_frames;
    if (elem_count != mel_f32.size())
    {
        throw std::runtime_error("writeMelSafetensors: tensor size mismatch");
    }
    std::vector<uint16_t> half(elem_count);
    for (size_t i = 0; i < elem_count; ++i) half[i] = f32_to_f16(mel_f32[i]);

    size_t const nbytes = half.size() * sizeof(uint16_t);
    Json header = Json{{"mel", Json{{"dtype", "F16"}, {"shape", Json::array({batch, n_mels, n_frames})},
                                  {"data_offsets", Json::array({0, nbytes})}}}};
    std::string header_str = header.dump();
    while ((header_str.size() % 8) != 0) header_str.push_back(' ');

    std::ofstream f(out_path, std::ios::binary);
    if (!f) throw std::runtime_error("writeMelSafetensors: cannot open " + out_path.string());
    uint64_t const header_len = static_cast<uint64_t>(header_str.size());
    f.write(reinterpret_cast<char const*>(&header_len), sizeof(header_len));
    f.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    f.write(reinterpret_cast<char const*>(half.data()), static_cast<std::streamsize>(nbytes));
}

//! Convert a cumulative float32-LE `pcm_b64` payload into a temp mel safetensors
//! file and return its path. Throws on malformed/too-short PCM. Caller is
//! responsible for removing the returned tempfile. Requires gMelExtractor.
std::filesystem::path pcmB64ToMelSafetensors(std::string const& pcmB64, std::string const& sid)
{
    std::vector<uint8_t> raw = base64Decode(pcmB64);
    if (raw.empty() || (raw.size() % sizeof(float)) != 0)
    {
        throw std::runtime_error("pcm_b64_malformed: decoded bytes not a positive multiple of 4 (float32-LE expected)");
    }
    std::vector<float> pcm(raw.size() / sizeof(float));
    std::memcpy(pcm.data(), raw.data(), raw.size());
    int32_t n_frames = 0;
    // Pad to the encoder chunk size so the [-1, 128, 100] static profile is
    // satisfied for short first hops (≈0.5 s → 50 frames → pad to 100).
    std::vector<float> mel = gMelExtractor->compute(pcm, &n_frames, kEncoderMelFramesPerChunk);
    if (n_frames <= 0)
    {
        throw std::runtime_error("pcm_too_short: produced 0 mel frames");
    }
    std::string safeId = sid.empty() ? std::string("s") : sid;
    for (auto& ch : safeId)
    {
        bool const ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'
            || ch == '-';
        if (!ok) ch = '_';
    }
    auto path = std::filesystem::temp_directory_path()
        / ("qwen3_asr_pcm_mel_" + safeId + "_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
    writeMelSafetensors(mel, gMelExtractor->n_mels(), n_frames, path);
    return path;
}

enum OptionId : int
{
    HELP = 1000,
    ENGINE_DIR,
    MULTIMODAL_ENGINE_DIR,
    MEL_SETTINGS,
    MEL_FILTERS,
    MAX_SLOTS,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --engineDir=<path> --multimodalEngineDir=<path>"
              << " [--melSettings=<json>] [--melFilters=<bin>] [--max_slots=<N>] [--debug]\n\n"
              << "Reads llm_inference-compatible JSON lines from stdin and writes JSON lines to stdout.\n"
              << "Pass --melSettings + --melFilters (or set EDGE_LLM_ASR_MEL_SETTINGS/EDGE_LLM_ASR_MEL_FILTERS)\n"
              << "to enable PCM-input streaming via `pcm_b64` chunk events.\n"
              << "--max_slots=<N> sets the ASR decoder slot-pool size (default 4; D1 concurrency).\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"engineDir", required_argument, 0, ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, MULTIMODAL_ENGINE_DIR},
        {"melSettings", required_argument, 0, MEL_SETTINGS},
        {"melFilters", required_argument, 0, MEL_FILTERS},
        {"max_slots", required_argument, 0, MAX_SLOTS},
        {"debug", no_argument, 0, DEBUG},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case ENGINE_DIR: args.engineDir = optarg; break;
        case MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case MEL_SETTINGS: args.melSettingsPath = optarg; break;
        case MEL_FILTERS: args.melFiltersPath = optarg; break;
        case MAX_SLOTS:
        {
            int const v = std::atoi(optarg);
            args.maxSlots = (v >= 1) ? v : 1; // clamp to >=1; 1 == single-session behavior
            break;
        }
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }
    // Env-var fallbacks let the worker auto-locate the assets shipped under
    // deploy/audio_preprocessing/ without forcing every caller to wire flags.
    if (args.melSettingsPath.empty())
    {
        if (char const* p = std::getenv("EDGE_LLM_ASR_MEL_SETTINGS")) args.melSettingsPath = p;
    }
    if (args.melFiltersPath.empty())
    {
        if (char const* p = std::getenv("EDGE_LLM_ASR_MEL_FILTERS")) args.melFiltersPath = p;
    }
    // D1 slot-pool size env fallback (matches the OVS EDGE_LLM_ASR_MAX_CONCURRENT
    // convention in spec §4). CLI --max_slots takes precedence when given.
    if (args.maxSlots == 4)
    {
        if (char const* p = std::getenv("EDGE_LLM_ASR_MAX_CONCURRENT"))
        {
            int const v = std::atoi(p);
            if (v >= 1) args.maxSlots = v;
        }
    }

    return !args.engineDir.empty() && !args.multimodalEngineDir.empty();
}

// Forward decl: the proven one-shot core (defined below) is reused by the
// streaming finalize path (handleEnd) which is defined earlier in the file.
Json runOneShotCore(Json input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap);

std::filesystem::path writeTempInput(Json const& input, std::string const& id)
{
    std::string safeId = id.empty() ? "request" : id;
    for (auto& ch : safeId)
    {
        bool const ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'
            || ch == '-';
        if (!ok)
        {
            ch = '_';
        }
    }
    auto path = std::filesystem::temp_directory_path()
        / ("qwen3_asr_worker_" + safeId + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
            + ".json");
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("Failed to open temp input file: " + path.string());
    }
    file << input.dump();
    return path;
}

// ===========================================================================
// Task #9: N>1 concurrent-session support (begin/chunk/end accumulation path).
// Task #10: true streaming PARTIALs (per-hop cumulative re-decode -> `partial`).
//
// Design (settled): each session reserves a LANE via SessionLaneManager (pure
// bookkeeping — lane reservation only, NOT AsrStreamingSessionRuntime KV-append).
// chunks accumulate the CUMULATIVE mel/pcm payload for the session AND run a
// cumulative re-decode on the audio-so-far, emitting an incremental `partial`
// per hop (task #10, first-cut: reuses the proven one-shot core -> guaranteed
// to converge to the one-shot transcript). At `end` (or a `last:true` chunk),
// the PROVEN one-shot core runs on the full cumulative audio under
// gEngineExecMutex, guaranteeing CER-0.0000 + byte-identity with the legacy
// one-shot path. The faster true incremental-KV per-hop append is a deferred
// optimization (see handleChunk note + plan §3).
//
// Concurrency model: the worker is single-threaded (one stdin loop). Sessions
// co-reside in gSessions and their lanes are held across interleaved
// begin/chunk/end events from multiple sids; the engine step itself is wrapped
// by gEngineExecMutex (kept for forward-compat with a future worker thread —
// single-threaded today so it never contends). A mandatory idle sweep releases
// lanes whose session went silent past kIdleTimeoutMs (half-open lane-leak guard,
// cf. the v2v slot-leak lesson).
// ===========================================================================

//! Per-session accumulation + lane bookkeeping.
struct SessionState
{
    int32_t laneId{-1};                       //!< Reserved lane (SessionLaneManager).
    std::string melPath;                      //!< Cumulative mel safetensors path (last chunk wins).
    std::string melPathOwned;                 //!< PCM-derived temp mel safetensors we own + must rm on release.
    std::string pcmB64;                       //!< Cumulative PCM (base64), if PCM-input path is used.
    Json beginMeta;                           //!< Begin-time metadata (sampling params etc.) to replay at finalize.
    std::chrono::steady_clock::time_point lastActivity{std::chrono::steady_clock::now()};
    //! Streaming-prefix path (OVS_ASR_STREAM_PREFIX=1, default OFF) state. When the
    //! flag is OFF these stay untouched and the cumulative re-decode path is
    //! byte-identical to #12. rawDecoded mirrors v0.7.x session.rawDecoded (the
    //! accumulated transcript WITH its leading "language <X>" tag, exactly as the
    //! model emits it). chunkId is the per-utterance hop counter that drives the
    //! unfixedChunkNum warmup (first N hops use NO prefix => full re-decode).
    std::string rawDecoded;                   //!< Accumulated transcript (lang tag included).
    int32_t chunkId{0};                       //!< Hop counter (mirrors v0.7.x chunk_id).
};

std::mutex gEngineExecMutex;                                   //!< Serializes the whole prefill+decode engine step.
std::unordered_map<std::string, SessionState> gSessions;       //!< sid -> session state.
std::unique_ptr<rt::SessionLaneManager> gLaneMgr;              //!< Lane allocator over the shared cache batch rows.
int32_t gMaxSlots{1};                                          //!< Advertised slot-pool size (== asrMax).
constexpr int64_t kIdleTimeoutMs = 30000;                      //!< Idle-session lane reclaim threshold.

//! Deterministic 63-bit ownerId from a session id (SessionLaneManager wants int64).
int64_t sidToOwnerId(std::string const& sid)
{
    return static_cast<int64_t>(std::hash<std::string>{}(sid) & 0x7fffffffffffffffLL);
}

//! Task #10: strip the leading "language <Lang>" tag the ASR head prepends, so
//! partials and finals expose the same bare transcript. Applied consistently to
//! both partial and final transcripts (idempotent — no tag => unchanged).
//! Case-insensitive ASCII compare of literal ``lit`` against ``s`` at ``pos``.
bool asciiIEqualsAt(std::string const& s, size_t pos, char const* lit)
{
    for (size_t i = 0; lit[i] != '\0'; ++i)
    {
        if (pos + i >= s.size()) return false;
        char a = s[pos + i];
        char b = lit[i];
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

//! Longest known ASR language name matching ``s`` at ``pos`` (0 = none). Used to
//! bound the "language <Lang>" strip so an ASCII transcript word glued to the tag
//! (e.g. "language EnglishGo home." — no <asr_text> separator) is NOT consumed.
size_t knownAsrLanguagePrefixLen(std::string const& s, size_t pos)
{
    static char const* const kLanguages[] = {
        "Chinese", "English", "Cantonese", "Japanese", "Korean", "French", "German", "Spanish",
        "Italian", "Portuguese", "Russian", "Arabic", "Hindi", "Bengali", "Urdu", "Indonesian",
        "Malay", "Vietnamese", "Thai", "Turkish", "Dutch", "Polish", "Ukrainian", "Swedish",
        "Norwegian", "Danish", "Finnish", "Greek", "Hebrew", "Persian", "Czech", "Slovak",
        "Hungarian", "Romanian", "Bulgarian", "Croatian", "Serbian", "Tamil", "Telugu",
        "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi", "Nepali", "Sinhala",
        "Burmese", "Khmer", "Lao", "Mongolian", "Tibetan", "Uyghur"};
    size_t best = 0;
    for (char const* lang : kLanguages)
    {
        size_t const n = std::strlen(lang);
        if (n > best && asciiIEqualsAt(s, pos, lang)) best = n;
    }
    return best;
}

std::string stripLangTag(std::string const& s)
{
    // Strip the "language <Lang>" / "language <Lang><asr_text>" head tag. The
    // <asr_text> token is often a special token decoded to empty, so the language
    // name can be glued to the first ASCII transcript word ("language EnglishGo
    // home."). Matching a KNOWN language name (not a greedy ASCII-letter run)
    // bounds the strip so the first word survives. CJK transcripts are non-ASCII
    // so they were already safe; unknown/absent tags leave ``s`` unchanged.
    static char const* kTag = "language ";
    static char const* kAsrTextTag = "<asr_text>";
    constexpr size_t kTagLen = 9; // strlen("language ")
    if (s.size() < kTagLen) return s;

    if (!asciiIEqualsAt(s, 0, kTag)) return s; // no "language " prefix -> untouched
    size_t j = kTagLen;

    // Forward-compat: if the <asr_text> separator survived, split on it directly.
    if (asciiIEqualsAt(s, j, kAsrTextTag))
    {
        return s.substr(j + std::strlen(kAsrTextTag));
    }

    size_t const langLen = knownAsrLanguagePrefixLen(s, j);
    if (langLen == 0) return s; // unknown language word -> don't risk eating text
    j += langLen;
    while (j < s.size() && (s[j] == ' ' || s[j] == '\t' || s[j] == '\r' || s[j] == '\n')) ++j;
    if (asciiIEqualsAt(s, j, kAsrTextTag)) j += std::strlen(kAsrTextTag);
    return s.substr(j);
}

//! Extract the bare (lang-tag-stripped) transcript from a runOneShotCore result.
//! Empty string if the core produced no output_text.
std::string transcriptFromCore(Json const& core)
{
    if (core.contains("responses") && core["responses"].is_array() && !core["responses"].empty())
    {
        Json const& r0 = core["responses"][0];
        if (r0.contains("output_text") && r0["output_text"].is_string())
        {
            return stripLangTag(r0["output_text"].get<std::string>());
        }
    }
    return std::string();
}

//! Global tokenizer handle for the streaming-prefix rollback (OVS_ASR_STREAM_PREFIX).
//! Loaded once in main() from the LLM engine dir (tokenizer.json + chat template)
//! when the flag is ON; null otherwise. Used ONLY to encode/decode the accumulated
//! transcript for the unfixedTokenNum roll-back (computePrefix). It is NOT used by
//! the runtime decode path, so the cumulative/one-shot golden path is unaffected.
std::unique_ptr<tokenizer::Tokenizer> gPrefixTokenizer;

// Forward declarations for the streaming-prefix path (definitions live after the
// asrStreamDelta namespace, alongside runOneShotCore which they reuse). handleChunk
// / handleEnd reference these, so declare them up here.
namespace asrStreamPrefix
{
struct HopOut
{
    bool ok{false};
    std::string generated;   //!< CONTINUATION text from the runtime (post lang-preamble strip).
    double totalMs{0.0};
    std::string error;
};
bool enabled();
std::string computePrefix(SessionState const& st, bool isFinish);
HopOut runPrefixHop(std::string const& sid, SessionState const& st, std::string const& prefix, int32_t capOverride,
    bool isFinish, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap);
} // namespace asrStreamPrefix

//! begin: reserve a lane (pool_saturated -> 4429), register session, ack.
void handleBegin(Json const& input)
{
    std::string const sid = input.value("id", "");
    if (sid.empty())
    {
        std::cout << Json{{"event", "error"}, {"ok", false}, {"error", "begin_missing_id"}}.dump() << std::endl;
        return;
    }
    if (gSessions.count(sid))
    {
        // Idempotent re-begin: ack the existing lane (no double-acquire).
        gSessions[sid].lastActivity = std::chrono::steady_clock::now();
        std::cout << Json{{"event", "begin_ack"}, {"id", sid}, {"lane", gSessions[sid].laneId}}.dump() << std::endl;
        return;
    }
    int32_t const lane = gLaneMgr->acquire(rt::LaneOwnerKind::kAsr, sidToOwnerId(sid));
    if (lane < 0)
    {
        // Pool saturated. NO worker restart — caller backs off / retries (HTTP 429
        // semantics, cf. the OVS session-limiter 4429 contract).
        std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "pool_saturated"},
            {"status", 4429}, {"max_slots", gMaxSlots}}.dump() << std::endl;
        return;
    }
    SessionState st;
    st.laneId = lane;
    st.lastActivity = std::chrono::steady_clock::now();
    // Capture sampling/meta knobs so finalize replays the SAME request envelope as
    // a one-shot call would (keeps byte-identity with the golden one-shot path).
    for (char const* k : {"batch_size", "temperature", "top_p", "top_k", "max_generate_length"})
    {
        if (input.contains(k)) st.beginMeta[k] = input[k];
    }
    gSessions.emplace(sid, std::move(st));
    std::cout << Json{{"event", "begin_ack"}, {"id", sid}, {"lane", lane}}.dump() << std::endl;
}

// Forward decls (handleChunk now finalizes on last:true via the handleEnd path).
void handleEnd(Json const& input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap);
Json buildFinalizeRequest(std::string const& sid, SessionState const& st);

//! Task #10: chunk = record the CUMULATIVE mel/pcm payload for this sid AND run a
//! cumulative re-decode (first-cut, guaranteed-converge: reuses the proven
//! one-shot core on the audio-so-far) to emit an incremental `partial`. On a
//! `last:true` chunk we route straight to the finalize path (emits `final`).
//!
//! DEFERRED OPTIMIZATION: the true incremental-KV per-hop append (decode only the
//! newly-arrived frames via AsrStreamingSessionRuntime::appendChunk /
//! decodeToTranscript on the session's lane, instead of re-decoding the whole
//! cumulative prefix every hop) is faster but not yet wired — see plan §3. This
//! cumulative re-decode is O(audio_so_far) per hop but is correct and converges
//! exactly to the one-shot transcript, so it ships first.
void handleChunk(Json const& input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    std::string const sid = input.value("id", "");
    auto it = gSessions.find(sid);
    if (it == gSessions.end())
    {
        std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "unknown_session"}}.dump()
                  << std::endl;
        return;
    }
    SessionState& st = it->second;
    st.lastActivity = std::chrono::steady_clock::now();
    // Cumulative semantics: each chunk carries the full-so-far audio descriptor
    // (the OVS accumulate driver re-points mel_path to the growing mel). Last wins.
    if (input.contains("mel_path") && input["mel_path"].is_string())
    {
        st.melPath = input["mel_path"].get<std::string>();
    }
    else if (input.contains("audio") && input["audio"].is_string())
    {
        st.melPath = input["audio"].get<std::string>();
    }
    if (input.contains("pcm_b64") && input["pcm_b64"].is_string())
    {
        st.pcmB64 = input["pcm_b64"].get<std::string>();
    }

    // `last:true` => this hop is the end-of-utterance: finalize via the proven path.
    bool const isLast = input.value("last", false);
    if (isLast)
    {
        handleEnd(input, runtime, stream, loraWeightsMap);
        return;
    }

    // PCM-input path (stream_mode=worker): the service sends raw float32-LE PCM in
    // `pcm_b64`. Convert the CUMULATIVE PCM to a mel safetensors here so the engine
    // request only ever carries a `.safetensors` audio path (requestFileParser
    // accepts nothing else → without this the parser threw on a missing `audio`
    // key and the worker SIGABRT'd; #11). Refuse cleanly if mel assets are absent.
    if (!st.pcmB64.empty())
    {
        if (!gMelExtractor)
        {
            std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "pcm_input_unsupported"},
                {"hint", "worker started without --melSettings/--melFilters; pass them or set "
                         "EDGE_LLM_ASR_MEL_{SETTINGS,FILTERS}"}}
                             .dump()
                      << std::endl;
            return;
        }
        try
        {
            if (!st.melPathOwned.empty())
            {
                std::error_code ec;
                std::filesystem::remove(st.melPathOwned, ec);
            }
            st.melPathOwned = pcmB64ToMelSafetensors(st.pcmB64, sid).string();
            st.melPath = st.melPathOwned;
        }
        catch (std::exception const& e)
        {
            std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "pcm_to_mel_failed"},
                {"detail", e.what()}}
                             .dump()
                      << std::endl;
            return;
        }
    }

    // No audio recorded yet => nothing to decode; just ack so the lane stays warm.
    if (st.melPath.empty() && st.pcmB64.empty())
    {
        std::cout << Json{{"event", "chunk_ack"}, {"id", sid}}.dump() << std::endl;
        return;
    }

    // Streaming-prefix path (OVS_ASR_STREAM_PREFIX=1): chunk-and-confirm rollback.
    // Each hop rolls back the last unfixedTokenNum tokens of the accumulated
    // transcript (computePrefix) and re-derives them + the new words from the
    // CURRENT cumulative audio, so a premature EOS / hallucinated tail in the
    // unstable window is REVISED instead of permanently committed. The first
    // unfixedChunkNum hops use NO prefix (full re-decode) while early audio is
    // unstable. Audio stays cumulative; only DECODE is flattened.
    if (asrStreamPrefix::enabled())
    {
        std::string const prefix = asrStreamPrefix::computePrefix(st, /*isFinish=*/false);
        asrStreamPrefix::HopOut hop;
        {
            std::lock_guard<std::mutex> lk(gEngineExecMutex);
            hop = asrStreamPrefix::runPrefixHop(
                sid, st, prefix, /*capOverride*/ -1, /*isFinish=*/false, runtime, stream, loraWeightsMap);
        }
        if (hop.ok)
        {
            // Re-derive the rolled-back tail: rawDecoded = confirmed prefix + fresh
            // continuation (NOT old-rawDecoded + continuation — that was the #25
            // poisoning bug). The unstable last-N tokens are thus replaced each hop.
            st.rawDecoded = prefix + hop.generated;
            st.chunkId += 1;
            std::cout << Json{{"event", "partial"}, {"id", sid}, {"text", stripLangTag(st.rawDecoded)}}.dump()
                      << std::endl;
        }
        else
        {
            Json ev = Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "partial_decode_failed"}};
            if (!hop.error.empty()) ev["detail"] = hop.error;
            std::cout << ev.dump() << std::endl;
        }
        return;
    }

    // Cumulative re-decode on the session's reserved lane (serialized by the engine
    // exec mutex, identical step to the one-shot/finalize path).
    Json const req = buildFinalizeRequest(sid, st);
    Json core;
    {
        std::lock_guard<std::mutex> lk(gEngineExecMutex);
        core = runOneShotCore(req, runtime, stream, loraWeightsMap);
    }
    if (core.value("ok", false))
    {
        std::cout << Json{{"event", "partial"}, {"id", sid}, {"text", transcriptFromCore(core)}}.dump() << std::endl;
    }
    else
    {
        // Decode failure on an intermediate hop is non-fatal to the session: surface
        // it but keep the lane so a later (longer) cumulative hop can still finalize.
        Json ev = Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "partial_decode_failed"}};
        if (core.contains("error")) ev["detail"] = core["error"];
        std::cout << ev.dump() << std::endl;
    }
}

//! Build the one-shot request envelope for a finalized session (cumulative audio).
Json buildFinalizeRequest(std::string const& sid, SessionState const& st)
{
    Json req = st.beginMeta.is_object() ? st.beginMeta : Json::object();
    if (!req.contains("batch_size")) req["batch_size"] = 1;
    if (!req.contains("temperature")) req["temperature"] = 1.0;
    if (!req.contains("top_p")) req["top_p"] = 1.0;
    if (!req.contains("top_k")) req["top_k"] = 1;
    if (!req.contains("max_generate_length")) req["max_generate_length"] = 256;
    req["id"] = sid;
    Json content = Json::array();
    // melPath is ALWAYS a `.safetensors` mel path here: mel_path callers set it
    // directly; pcm_b64 callers had it converted to a temp mel safetensors in
    // handleChunk/handleEnd (st.melPathOwned). requestFileParser accepts ONLY a
    // .safetensors audio path, so we never emit raw pcm_b64 into the request.
    if (!st.melPath.empty())
    {
        content.push_back(Json{{"type", "audio"}, {"audio", st.melPath}});
    }
    req["requests"] = Json::array({Json{{"messages", Json::array({Json{{"role", "user"}, {"content", content}}})},
        {"reference", "ref"}}});
    return req;
}

//! end / last:true: finalize via the PROVEN one-shot core, release lane, emit final.
void handleEnd(Json const& input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    std::string const sid = input.value("id", "");
    auto it = gSessions.find(sid);
    if (it == gSessions.end())
    {
        std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "unknown_session"}}.dump()
                  << std::endl;
        return;
    }
    SessionState st = it->second; // copy; we release the slot below regardless of outcome
    // Allow `end` to carry the final cumulative audio too.
    if (input.contains("mel_path") && input["mel_path"].is_string()) st.melPath = input["mel_path"].get<std::string>();
    else if (input.contains("audio") && input["audio"].is_string()) st.melPath = input["audio"].get<std::string>();
    bool endCarriedPcm = false;
    if (input.contains("pcm_b64") && input["pcm_b64"].is_string())
    {
        st.pcmB64 = input["pcm_b64"].get<std::string>();
        endCarriedPcm = true;
    }

    // If `end` itself carried fresh cumulative PCM, convert it to a mel safetensors
    // now (chunks already converted theirs into st.melPath/melPathOwned). Same guard
    // + conversion as handleChunk so the engine request only ever sees a .safetensors.
    Json finalEv;
    if (endCarriedPcm && gMelExtractor)
    {
        try
        {
            if (!st.melPathOwned.empty())
            {
                std::error_code ec;
                std::filesystem::remove(st.melPathOwned, ec);
            }
            st.melPathOwned = pcmB64ToMelSafetensors(st.pcmB64, sid).string();
            st.melPath = st.melPathOwned;
        }
        catch (std::exception const& e)
        {
            st.melPath.clear();
            finalEv = Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "pcm_to_mel_failed"},
                {"detail", e.what()}};
        }
    }
    else if (endCarriedPcm && !gMelExtractor)
    {
        finalEv = Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "pcm_input_unsupported"},
            {"hint", "worker started without --melSettings/--melFilters"}};
    }

    if (!finalEv.is_null())
    {
        // pcm-conversion error already populated finalEv; fall through to release.
    }
    else if (st.melPath.empty())
    {
        finalEv = Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "no_audio_accumulated"}};
    }
    else if (asrStreamPrefix::enabled())
    {
        // Streaming-prefix finalize. Partials (non-final hops) stay the cheap
        // rollback-continuation. But the FINAL transcript must be BYTE-EXACT to the
        // deployed one-shot, so by default we re-run the PROVEN golden one-shot core
        // (buildFinalizeRequest = user-only message over the FULL cumulative audio,
        // NO assistant-prefix) over the cumulative audio ONCE here — exactly what the
        // deployed/default path (the `else` branch below) does. This eliminates the
        // CER 0.01-0.025 rollback residual while keeping partials flat/cheap.
        // OVS_ASR_STREAM_PREFIX_FINAL_ONESHOT=0 reverts to the old rollback-final hop.
        bool finalOneShot = true;
        if (char const* fp = std::getenv("OVS_ASR_STREAM_PREFIX_FINAL_ONESHOT"))
            finalOneShot = !(fp[0] == '0' && fp[1] == '\0');
        if (finalOneShot)
        {
            Json const req = buildFinalizeRequest(sid, st);
            Json core;
            {
                std::lock_guard<std::mutex> lk(gEngineExecMutex);
                core = runOneShotCore(req, runtime, stream, loraWeightsMap);
            }
            finalEv = Json{{"event", "final"}, {"id", sid}, {"ok", core.value("ok", false)}};
            if (core.contains("responses")) finalEv["responses"] = core["responses"];
            if (core.value("ok", false)) finalEv["text"] = transcriptFromCore(core);
            if (core.contains("total_ms")) finalEv["total_ms"] = core["total_ms"];
            if (core.contains("error")) finalEv["error"] = core["error"];
        }
        else
        {
            // Legacy rollback-final: one last UNCAPPED chunk-and-confirm hop over the
            // FULL cumulative audio (isFinish=true keeps >=1 confirmed prefix token).
            std::string const prefix = asrStreamPrefix::computePrefix(st, /*isFinish=*/true);
            asrStreamPrefix::HopOut hop;
            {
                std::lock_guard<std::mutex> lk(gEngineExecMutex);
                hop = asrStreamPrefix::runPrefixHop(
                    sid, st, prefix, /*uncapped*/ -1, /*isFinish=*/true, runtime, stream, loraWeightsMap);
            }
            std::string const finalRaw = prefix + hop.generated;
            finalEv = Json{{"event", "final"}, {"id", sid}, {"ok", hop.ok}};
            if (hop.ok)
            {
                finalEv["responses"] = Json::array(
                    {Json{{"request_idx", 0}, {"batch_idx", 0}, {"output_text", finalRaw}}});
                finalEv["text"] = stripLangTag(finalRaw);
            }
            finalEv["total_ms"] = hop.totalMs;
            if (!hop.error.empty()) finalEv["error"] = hop.error;
        }
    }
    else
    {
        Json const req = buildFinalizeRequest(sid, st);
        Json core;
        {
            std::lock_guard<std::mutex> lk(gEngineExecMutex);
            core = runOneShotCore(req, runtime, stream, loraWeightsMap);
        }
        // Re-shape the one-shot `done` into a streaming `final` (same output_text).
        // Also expose a lang-tag-stripped `text` so partial/final agree byte-for-byte
        // on the bare transcript (task #10 convergence contract).
        finalEv = Json{{"event", "final"}, {"id", sid}, {"ok", core.value("ok", false)}};
        if (core.contains("responses")) finalEv["responses"] = core["responses"];
        if (core.value("ok", false)) finalEv["text"] = transcriptFromCore(core);
        if (core.contains("total_ms")) finalEv["total_ms"] = core["total_ms"];
        if (core.contains("error")) finalEv["error"] = core["error"];
    }

    // Remove any PCM-derived temp mel safetensors we own. Both the original
    // session's last-chunk tempfile (it->second.melPathOwned) and a fresh one the
    // `end` event may have produced into the local copy (st.melPathOwned) get rm'd.
    auto rmOwned = [](std::string const& p) {
        if (!p.empty())
        {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    };
    rmOwned(it->second.melPathOwned);
    if (st.melPathOwned != it->second.melPathOwned) rmOwned(st.melPathOwned);

    // Release the lane back to the pool (guaranteed, success or failure) and drop the session.
    gLaneMgr->release(st.laneId);
    gSessions.erase(sid);
    std::cout << finalEv.dump() << std::endl;
}

//! Mandatory idle-sweep: reclaim lanes of sessions silent past kIdleTimeoutMs.
void sweepIdleSessions()
{
    if (gSessions.empty()) return;
    auto const now = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, int32_t>> reap;
    for (auto const& [sid, st] : gSessions)
    {
        int64_t const idleMs
            = std::chrono::duration_cast<std::chrono::milliseconds>(now - st.lastActivity).count();
        if (idleMs > kIdleTimeoutMs)
        {
            reap.emplace_back(sid, st.laneId);
        }
    }
    for (auto const& [sid, lane] : reap)
    {
        auto sit = gSessions.find(sid);
        if (sit != gSessions.end() && !sit->second.melPathOwned.empty())
        {
            std::error_code ec;
            std::filesystem::remove(sit->second.melPathOwned, ec);
        }
        gLaneMgr->release(lane);
        gSessions.erase(sid);
        std::cout << Json{{"event", "timeout"}, {"id", sid}, {"ok", false}, {"error", "idle_timeout"},
            {"idle_ms", kIdleTimeoutMs}}.dump() << std::endl;
    }
}

// ===========================================================================
// Phase 2a: alternate finalize path via AsrStreamingSessionRuntime (single hop).
//
// Gated by env OVS_ASR_SESSION_PATH=1 (default OFF -> runOneShotCore unchanged).
// When ON, the FULL utterance is driven through the streaming-session runtime as
// a SINGLE chunk (mirrors examples/llm/spike_v080_m6_audio_streaming.cpp --chunks 1):
//   beginAsrSession -> appendChunk(full mel, isFinal=true) -> decodeToTranscript
//   -> getTranscript -> endAsrSession.
// The mel consumed is the SAME `.safetensors` the one-shot path would feed via
// requestFileParser (extracted from the request envelope below), and the audio-pad
// token layout matches the one-shot prompt exactly (Qwen3-ASR config.json):
//   prefix = <|im_start|>user\n      = [151644, 872, 198]
//   audio  = <|audio_start|> Nx<|audio_pad|> <|audio_end|> = [151669, 151676*N, 151670]
//   suffix = <|im_end|>\n<|im_start|>assistant\n = [151645, 198, 151644, 77091, 198]
// (audio_*_token_id read from engines-v080/llm/config.json: start=151669, pad=151676,
//  end=151670; chat template default_system_prompt is empty -> no system block, just
//  like the one-shot user-message prompt.) N is derived from the REAL audio by probing
// the encoder (encodeAudioChunk -> outEmbedding.getShape()[0]), never hardcoded.
//
// This is OFFLINE single-hop EQUIVALENCE validation only (Phase 2a). Delta/multi-hop
// /partial extraction is Phase 2b.
// ===========================================================================
namespace asrSessionPath
{
constexpr int32_t kImStart = 151644;
constexpr int32_t kUser = 872;
constexpr int32_t kNl = 198;
constexpr int32_t kAudioStart = 151669;
constexpr int32_t kAudioPad = 151676;
constexpr int32_t kAudioEnd = 151670;
constexpr int32_t kImEnd = 151645;
constexpr int32_t kAssistant = 77091;
constexpr int32_t kMaxAudioTokensPerChunk = 2048; // worst-case per-chunk scratch size (matches m6).

//! True iff OVS_ASR_SESSION_PATH=1 (cached once; default OFF). When OFF the worker
//! is byte-identical to v080-0021 #10 — runOneShotCore is the unchanged golden path.
bool enabled()
{
    static bool const on = []() {
        char const* p = std::getenv("OVS_ASR_SESSION_PATH");
        return p != nullptr && std::string(p) == "1";
    }();
    return on;
}

//! Dig the audio mel `.safetensors` path out of the one-shot request envelope
//! (requests[0].messages[*].content[*] with type=="audio"). Returns "" if none.
std::string extractMelPath(Json const& input)
{
    if (!input.contains("requests") || !input["requests"].is_array() || input["requests"].empty())
        return std::string();
    Json const& r0 = input["requests"][0];
    if (!r0.contains("messages") || !r0["messages"].is_array())
        return std::string();
    for (Json const& msg : r0["messages"])
    {
        if (!msg.contains("content") || !msg["content"].is_array())
            continue;
        for (Json const& c : msg["content"])
        {
            if (c.contains("type") && c["type"] == "audio" && c.contains("audio") && c["audio"].is_string())
                return c["audio"].get<std::string>();
        }
    }
    return std::string();
}

//! Drive the full mel through AsrStreamingSessionRuntime as a SINGLE hop and return
//! the bare transcript. Throws on any session-runtime failure. activeBatchSize=1.
std::string runSingleHop(std::string const& melPath, rt::LLMInferenceRuntime& runtime, int32_t maxGenerateLength,
    cudaStream_t stream)
{
    // 1) Probe the encoder to learn N (== number of <|audio_pad|> tokens this mel yields).
    rt::Tensor probeScratch;
    if (!runtime.allocateChunkAudioEmbedding(probeScratch, kMaxAudioTokensPerChunk))
        throw std::runtime_error("session_path: allocateChunkAudioEmbedding failed");
    rt::audioUtils::AudioData melDesc;
    melDesc.melSpectrogramPath = melPath;
    melDesc.melSpectrogramFormat = "safetensors";
    if (!runtime.encodeAudioChunk(melDesc, probeScratch, stream))
        throw std::runtime_error("session_path: encodeAudioChunk (probe) failed for " + melPath);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int32_t const nTokens = static_cast<int32_t>(probeScratch.getShape()[0]);
    if (nTokens <= 0)
        throw std::runtime_error("session_path: probe produced 0 audio tokens");

    // 2) maxPositions covers prefix(4) + N pads + suffix(6) + slack (matches m6: N + 64).
    int32_t const maxPositions = nTokens + 64;

    // 3) Single-session over lane 0 (Phase 3 single-lane; activeBatchSize == 1).
    rt::AsrStreamingSessionRuntime sess(runtime, /*lane*/ 0);
    if (!sess.beginAsrSession(/*promptTokenIds*/ {}, maxPositions, kMaxAudioTokensPerChunk, stream, maxGenerateLength))
        throw std::runtime_error("session_path: beginAsrSession failed");

    // 4) SINGLE chunk = the whole audio. Carries the full prompt: prefix + audio_start,
    //    then N audio pads, then audio_end + assistant suffix; isFinal=true samples a token.
    std::vector<int32_t> slice;
    slice.push_back(kImStart);
    slice.push_back(kUser);
    slice.push_back(kNl);
    slice.push_back(kAudioStart);
    for (int32_t k = 0; k < nTokens; ++k)
        slice.push_back(kAudioPad);
    slice.push_back(kAudioEnd);
    slice.push_back(kImEnd);
    slice.push_back(kNl);
    slice.push_back(kImStart);
    slice.push_back(kAssistant);
    slice.push_back(kNl);

    rt::AudioChunk chunk;
    chunk.data.melSpectrogramPath = melPath;
    chunk.data.melSpectrogramFormat = "safetensors";
    chunk.hasAudio = true;
    if (!sess.appendChunk(slice, chunk, /*isFinalChunk*/ true, stream))
    {
        sess.endAsrSession(stream);
        throw std::runtime_error("session_path: appendChunk (final) failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (sess.status() != rt::AsrSessionStatus::kFinished)
    {
        sess.endAsrSession(stream);
        throw std::runtime_error("session_path: session not finished after final chunk");
    }

    // 5) Decode-to-EOS, read transcript, then end the session (full-batch reset, Phase 3).
    if (!sess.decodeToTranscript(stream))
    {
        sess.endAsrSession(stream);
        throw std::runtime_error("session_path: decodeToTranscript failed");
    }
    std::string const transcript = sess.getTranscript();
    sess.endAsrSession(stream);
    return transcript;
}

//! Session-path analogue of runOneShotCore: same input/output envelope, but the
//! transcript is produced by the streaming-session single-hop instead of
//! handleRequest. Returns the SAME Json shape so all callers / the A/B gate compare
//! apples-to-apples. The output_text carries the bare transcript exactly as the
//! audio head emits it (lang tag included) — identical to handleRequest's outputTexts.
Json runSessionCore(Json input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream)
{
    Json response;
    auto const requestStart = std::chrono::steady_clock::now();
    try
    {
        std::string const id = input.value("id", "");
        int64_t const maxGenLenOverride = input.value("max_generate_length_override", -1);
        int64_t maxGenLen = input.value("max_generate_length", 256);
        if (maxGenLenOverride > 0) maxGenLen = maxGenLenOverride;
        if (maxGenLen <= 0) maxGenLen = 256;

        std::string const melPath = extractMelPath(input);
        if (melPath.empty())
            throw std::runtime_error("session_path: no audio mel path in request");

        std::string const text = runSingleHop(melPath, runtime, static_cast<int32_t>(maxGenLen), stream);

        Json responses = Json::array();
        responses.push_back(Json{{"request_idx", 0}, {"batch_idx", 0}, {"output_text", text}});
        double const totalMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - requestStart).count();
        response = Json{{"id", id}, {"event", "done"}, {"ok", true}, {"responses", responses},
            {"total_ms", totalMs}, {"path", "session"}};
    }
    catch (std::exception const& e)
    {
        response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}, {"path", "session"}};
    }
    return response;
}
} // namespace asrSessionPath

// ===========================================================================
// Phase 2b: MULTI-HOP DELTA streaming path with non-evicting peek partials.
//
// Gated by env OVS_ASR_STREAM_DELTA=1 (default OFF; orthogonal to OVS_ASR_SESSION_PATH).
// Splits a clip's mel into K disjoint hops (the .cN delta slices from gen_mel_np.py) and
// drives them through AsrStreamingSessionRuntime as a TRUE incremental-KV stream:
//   begin(prompt-open: prefix + audio_start, NO audio_end/suffix yet)
//   hop k<K-1 (non-final): appendChunk(delta pads + delta mel, isFinal=false)  [O(delta) work]
//                          -> peekTranscriptWithSuffix(...) -> emit `partial`   [non-evicting]
//   hop K-1   (final):     appendChunk(last delta pads + audio_end + suffix, isFinal=true)
//                          -> decodeToTranscript -> `final` (REAL, no restore) -> endAsrSession
//
// The (encode+append) wall time per hop is logged SEPARATELY from the peek-decode time so the
// O(N) (flat per-hop append) vs O(N^2) (cumulative re-decode) win is observable (G2). Per-hop
// pre-peek vs post-peek KV length + tokenIds + activeBatch are captured to prove zero peek
// residue across interleaved appends (G3). The FINAL transcript must be byte-identical to the
// one-shot golden (G1).
//
// Single-token guard (v080-0004): a non-final delta whose audio_pad count would be 1 is BUFFERED
// and merged into the next hop (audio deltas are ~>=100 pads in practice, but guarded anyway).
// ===========================================================================
namespace asrStreamDelta
{
using namespace asrSessionPath; // reuse the token-id constants + kMaxAudioTokensPerChunk.

//! True iff OVS_ASR_STREAM_DELTA=1 (cached once; default OFF).
bool enabled()
{
    static bool const on = []() {
        char const* p = std::getenv("OVS_ASR_STREAM_DELTA");
        return p != nullptr && std::string(p) == "1";
    }();
    return on;
}

//! Probe one delta mel through the encoder to learn its audio-token count (nk pads).
int32_t probeDelta(rt::LLMInferenceRuntime& runtime, std::string const& melPath, rt::Tensor& scratch,
    cudaStream_t stream)
{
    rt::audioUtils::AudioData mel;
    mel.melSpectrogramPath = melPath;
    mel.melSpectrogramFormat = "safetensors";
    if (!runtime.encodeAudioChunk(mel, scratch, stream))
        return -1;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<int32_t>(scratch.getShape()[0]);
}

//! Drive the multi-hop delta stream. Emits per-hop `partial` + `hop_metric` events on `out`,
//! returns the FINAL bare transcript (lang tag included; same shape as runSessionCore).
//! `deltaMels` are the K disjoint .cN mel slices, in order.
Json runDelta(std::vector<std::string> const& deltaMels, rt::LLMInferenceRuntime& runtime,
    int32_t maxGenLen, cudaStream_t stream, std::string const& id)
{
    auto const reqStart = std::chrono::steady_clock::now();
    Json perHop = Json::array();

    // 1) Probe every delta to learn nk, total tokens, and to apply the single-token guard.
    rt::Tensor probeScratch;
    if (!runtime.allocateChunkAudioEmbedding(probeScratch, kMaxAudioTokensPerChunk))
        return Json{{"event", "error"}, {"ok", false}, {"error", "delta: allocateChunkAudioEmbedding failed"}};
    int32_t const K = static_cast<int32_t>(deltaMels.size());
    std::vector<int32_t> nTok(K, 0);
    int32_t total = 0;
    for (int32_t i = 0; i < K; ++i)
    {
        nTok[i] = probeDelta(runtime, deltaMels[i], probeScratch, stream);
        if (nTok[i] < 0)
            return Json{{"event", "error"}, {"ok", false}, {"error", "delta: probe failed for " + deltaMels[i]}};
        total += nTok[i];
    }
    int32_t const maxPositions = total + 64;

    // 2) begin (prompt-open only: prefix + audio_start). No audio_end / suffix yet.
    rt::AsrStreamingSessionRuntime sess(runtime, /*lane*/ 0);
    if (!sess.beginAsrSession(/*promptTokenIds*/ {}, maxPositions, kMaxAudioTokensPerChunk, stream, maxGenLen))
        return Json{{"event", "error"}, {"ok", false}, {"error", "delta: beginAsrSession failed"}};

    int32_t const lane = 0;
    int32_t const lastIdx = K - 1;
    std::vector<int32_t> const suffix
        = {kAudioEnd, kImEnd, kNl, kImStart, kAssistant, kNl}; // speculative + real suffix.

    int32_t bufferedPads = 0; // single-token guard carry (count of audio pads merged into next hop).

    for (int32_t i = 0; i <= lastIdx; ++i)
    {
        bool const isFinal = (i == lastIdx);
        int32_t padCount = nTok[i] + bufferedPads;

        // Single-token guard: never let a NON-FINAL append be exactly 1 audio pad -> buffer + merge.
        if (!isFinal && padCount == 1)
        {
            bufferedPads = padCount;
            perHop.push_back(Json{{"hop", i}, {"buffered", true}, {"pads", nTok[i]},
                {"note", "single-token-guard: deferred to next hop"}});
            continue;
        }
        bufferedPads = 0;

        // Build this hop's token slice.
        std::vector<int32_t> slice;
        if (i == 0)
        {
            slice.push_back(kImStart);
            slice.push_back(kUser);
            slice.push_back(kNl);
            slice.push_back(kAudioStart);
        }
        for (int32_t k = 0; k < padCount; ++k)
            slice.push_back(kAudioPad);
        if (isFinal)
            for (int32_t t : suffix)
                slice.push_back(t);

        rt::AudioChunk chunk;
        chunk.data.melSpectrogramPath = deltaMels[i];
        chunk.data.melSpectrogramFormat = "safetensors";
        chunk.hasAudio = true;

        // ── G2: time (encodeMelChunk + appendChunk) ONLY (the O(N) work). ──
        auto const t0 = std::chrono::steady_clock::now();
        bool const ok = sess.appendChunk(slice, chunk, isFinal, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const t1 = std::chrono::steady_clock::now();
        double const appendMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!ok)
        {
            sess.endAsrSession(stream);
            return Json{{"event", "error"}, {"ok", false},
                {"error", "delta: appendChunk hop " + std::to_string(i) + " failed"}};
        }

        if (!isFinal)
        {
            // ── G3: pre-peek durable state snapshot. ──
            auto const& ctxPre = sess.contextForTesting();
            int32_t const kvPre = sess.peekKvCacheLength(stream);
            std::vector<int32_t> const tokPre = ctxPre.tokenIds.at(lane);
            int32_t const abPre = ctxPre.activeBatchSize;

            // ── PEEK-WITH-SUFFIX (the CRUX): non-evicting partial from a MID-AUDIO session. ──
            auto const p0 = std::chrono::steady_clock::now();
            rt::DecodePeekResult pr = sess.peekTranscriptWithSuffix(suffix, maxGenLen, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto const p1 = std::chrono::steady_clock::now();
            double const peekMs = std::chrono::duration<double, std::milli>(p1 - p0).count();

            // ── G3: post-peek durable state (must equal pre-peek -> zero residue). ──
            auto const& ctxPost = sess.contextForTesting();
            int32_t const kvPost = sess.peekKvCacheLength(stream);
            std::vector<int32_t> const tokPost = ctxPost.tokenIds.at(lane);
            int32_t const abPost = ctxPost.activeBatchSize;
            bool const residueFree = (kvPost == kvPre) && (tokPost == tokPre) && (abPost == abPre);

            std::string const partial = stripLangTag(pr.transcript);
            std::cout << Json{{"event", "partial"}, {"id", id}, {"hop", i}, {"text", partial}}.dump()
                      << std::endl;

            perHop.push_back(Json{{"hop", i}, {"final", false}, {"pads", padCount}, {"kv", kvPre},
                {"append_ms", appendMs}, {"peek_ms", peekMs}, {"peek_ok", pr.ok},
                {"partial", partial},
                {"g3_kv_pre", kvPre}, {"g3_kv_post", kvPost}, {"g3_tok_pre", tokPre.size()},
                {"g3_tok_post", tokPost.size()}, {"g3_tok_eq", (tokPost == tokPre)},
                {"g3_ab_pre", abPre}, {"g3_ab_post", abPost}, {"g3_residue_free", residueFree}});
        }
        else
        {
            int32_t const kvFinal = sess.peekKvCacheLength(stream);
            perHop.push_back(Json{{"hop", i}, {"final", true}, {"pads", padCount}, {"kv", kvFinal},
                {"append_ms", appendMs}});
        }
    }

    // 3) Final REAL decode (no restore) -> transcript -> end.
    if (sess.status() != rt::AsrSessionStatus::kFinished)
    {
        sess.endAsrSession(stream);
        return Json{{"event", "error"}, {"ok", false}, {"error", "delta: not finished after final hop"}};
    }
    if (!sess.decodeToTranscript(stream))
    {
        sess.endAsrSession(stream);
        return Json{{"event", "error"}, {"ok", false}, {"error", "delta: final decodeToTranscript failed"}};
    }
    std::string const finalText = sess.getTranscript();
    sess.endAsrSession(stream);

    double const totalMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reqStart).count();
    Json responses = Json::array();
    responses.push_back(Json{{"request_idx", 0}, {"batch_idx", 0}, {"output_text", finalText}});
    return Json{{"id", id}, {"event", "done"}, {"ok", true}, {"responses", responses},
        {"total_ms", totalMs}, {"path", "stream_delta"}, {"per_hop", perHop}};
}

//! Dig the delta mel list out of the request (top-level "delta_mels": [paths...]).
std::vector<std::string> extractDeltaMels(Json const& input)
{
    std::vector<std::string> out;
    if (input.contains("delta_mels") && input["delta_mels"].is_array())
        for (Json const& m : input["delta_mels"])
            if (m.is_string())
                out.push_back(m.get<std::string>());
    return out;
}

//! Session-delta analogue of runSessionCore: same envelope, multi-hop delta path.
Json runDeltaCore(Json input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream)
{
    try
    {
        std::string const id = input.value("id", "");
        int64_t maxGenLen = input.value("max_generate_length", 256);
        int64_t const ovr = input.value("max_generate_length_override", -1);
        if (ovr > 0) maxGenLen = ovr;
        if (maxGenLen <= 0) maxGenLen = 256;

        std::vector<std::string> deltas = extractDeltaMels(input);
        if (deltas.empty())
            return Json{{"event", "error"}, {"ok", false},
                {"error", "stream_delta: no delta_mels[] in request"}, {"path", "stream_delta"}};
        return runDelta(deltas, runtime, static_cast<int32_t>(maxGenLen), stream, id);
    }
    catch (std::exception const& e)
    {
        return Json{{"event", "error"}, {"ok", false}, {"error", e.what()}, {"path", "stream_delta"}};
    }
}
} // namespace asrStreamDelta

// ===========================================================================
// Streaming-prefix CHUNK-AND-CONFIRM ROLLBACK (v0.7.x flat-decode, faithful port).
//
// Gated by env OVS_ASR_STREAM_PREFIX=1 (default OFF -> handleChunk/handleEnd take
// the cumulative re-decode path, byte-identical to #12). Mirrors the v0.7.x worker
// (asr-worker-build-verify/qwen3_asr_worker.cpp): session params unfixedChunkNum=2,
// unfixedTokenNum=5, maxDecodeTokensPerHop=64; computePrefix() / runStreamingHop().
//
// Mechanism (THE fix the #25 attempt dropped): each hop the accumulated transcript
// is rolled back by unfixedTokenNum tokens — the last N tokens are treated as
// UNCONFIRMED/revisable. The runtime re-derives them (plus the new words) from the
// CURRENT cumulative audio via an assistant-prefix continuation. A premature EOS /
// hallucinated tail in that unstable window is therefore REVISED with more audio,
// never permanently committed. Only the confirmed prefix (all but the last N
// tokens) is reused cheaply (prefilled, not re-decoded). The first unfixedChunkNum
// hops use NO prefix (full re-decode) while early audio is unstable.
//
// Audio stays CUMULATIVE (full mel each hop) — this flattens DECODE, not encode.
// Per hop:
//   prefix = computePrefix(rawDecoded, chunkId)   // rolled-back confirmed text
//   request = user(audio=full cumulative mel) + assistant(prefix)   // open turn
//   cap = maxDecodeTokensPerHop (final hop uncapped)
//   rawDecoded = prefix + parseAsrText(generated) // re-derive the rolled-back tail
// ===========================================================================
namespace asrStreamPrefix
{
constexpr char const* kUtf8Replacement = "\xEF\xBF\xBD"; //!< U+FFFD as UTF-8 (multibyte-split guard).

//! True iff OVS_ASR_STREAM_PREFIX=1 (cached once; default OFF).
bool enabled()
{
    static bool const on = []() {
        char const* p = std::getenv("OVS_ASR_STREAM_PREFIX");
        return p != nullptr && std::string(p) == "1";
    }();
    return on;
}

//! unfixedChunkNum (mirrors v0.7.x=2): first N hops use NO prefix => full re-decode
//! (early audio is unstable, do not confirm anything yet). Env: OVS_ASR_PREFIX_WARMUP.
int32_t unfixedChunkNum()
{
    static int32_t const v = []() {
        char const* p = std::getenv("OVS_ASR_PREFIX_WARMUP");
        int32_t x = (p != nullptr) ? std::atoi(p) : 2;
        return (x >= 0) ? x : 2;
    }();
    return v;
}

//! unfixedTokenNum (mirrors v0.7.x=5): roll back the last N tokens of the accumulated
//! transcript each hop (treat them UNCONFIRMED/revisable). Env: OVS_ASR_PREFIX_UNFIXED
//! (the G-correct tuning knob; sweep {5,8,12}).
int32_t unfixedTokenNum()
{
    static int32_t const v = []() {
        char const* p = std::getenv("OVS_ASR_PREFIX_UNFIXED");
        int32_t x = (p != nullptr) ? std::atoi(p) : 5;
        return (x >= 1) ? x : 5;
    }();
    return v;
}

//! maxDecodeTokensPerHop (mirrors v0.7.x=64): per-hop decode cap for non-final hops.
//! Env: OVS_ASR_STREAM_PREFIX_CAP. Final hop overrides this with an uncapped decode.
int32_t maxDecodeTokensPerHop()
{
    static int32_t const v = []() {
        char const* p = std::getenv("OVS_ASR_STREAM_PREFIX_CAP");
        int32_t x = (p != nullptr) ? std::atoi(p) : 64;
        return (x >= 1) ? x : 64;
    }();
    return v;
}

//! Strip a leading "language X<asr_text>" / "language Xxx " preamble from a raw
//! model output (mirrors v0.7.x parseAsrText). On a CONTINUATION hop the model does
//! not re-emit the tag, so this is a no-op; on a fresh (warmup) hop it strips it.
std::string parseAsrText(std::string const& raw)
{
    static constexpr char const* kAsrTextTag = "<asr_text>";
    auto tagPos = raw.find(kAsrTextTag);
    if (tagPos != std::string::npos)
    {
        return raw.substr(tagPos + std::strlen(kAsrTextTag));
    }
    // No tag: reuse the unified, known-language-bounded strip (handles the glued
    // "language EnglishGo home." case without eating the first transcript word).
    return stripLangTag(raw);
}

//! THE rollback (faithful port of v0.7.x computePrefix). Returns the CONFIRMED
//! prefix = accumulated transcript with its last unfixedTokenNum tokens dropped
//! (those are re-derived from current audio this hop). Returns "" for the first
//! unfixedChunkNum hops (warmup: no prefix, full re-decode). UTF-8 guard: if the
//! truncated decode would split a multibyte char (U+FFFD appears), increment the
//! roll-back count and retry so we never cut a CJK character mid-byte.
std::string computePrefix(SessionState const& st, bool isFinish)
{
    tokenizer::Tokenizer* tok = gPrefixTokenizer.get();
    if (tok == nullptr || st.rawDecoded.empty())
    {
        return "";
    }
    if (st.chunkId < unfixedChunkNum())
    {
        return "";
    }
    auto tokens = tok->encode(st.rawDecoded, /*addBos=*/false, /*addEos=*/false);
    if (tokens.empty())
    {
        return "";
    }
    int k = unfixedTokenNum();
    int const total = static_cast<int>(tokens.size());
    while (true)
    {
        int end = total - k;
        if (isFinish)
        {
            end = std::max(1, end);
        }
        else
        {
            end = std::max(0, end);
        }
        std::string prefix;
        if (end > 0)
        {
            std::vector<tokenizer::Rank> slice(tokens.begin(), tokens.begin() + end);
            prefix = tok->decode(slice, /*skipSpecialTokens=*/false);
        }
        if (prefix.find(kUtf8Replacement) == std::string::npos)
        {
            return prefix;
        }
        if (!isFinish && end == 0)
        {
            return "";
        }
        if (isFinish && end == 1)
        {
            return prefix;
        }
        ++k; // multibyte char split at the boundary -> roll back one more token.
    }
}

//! Build the one-shot-style request for an assistant-prefix continuation hop. The
//! user message carries the FULL cumulative audio; the assistant message carries the
//! rolled-back `prefix`. apply_chat_template=true + add_generation_prompt=false
//! leave the assistant turn OPEN so the runtime generates ONLY the continuation
//! (= the rolled-back tail re-derived from current audio + any new words). When
//! prefix is empty (warmup hops) we open the turn with add_generation_prompt=true
//! and NO assistant message (a user-only message with add_generation_prompt=false
//! does not open the turn). capOverride is plumbed as max_generate_length_override.
Json buildPrefixRequest(std::string const& sid, SessionState const& st, std::string const& prefix, int32_t capOverride)
{
    Json req = st.beginMeta.is_object() ? st.beginMeta : Json::object();
    if (!req.contains("batch_size")) req["batch_size"] = 1;
    if (!req.contains("temperature")) req["temperature"] = 1.0;
    if (!req.contains("top_p")) req["top_p"] = 1.0;
    if (!req.contains("top_k")) req["top_k"] = 1;
    if (!req.contains("max_generate_length")) req["max_generate_length"] = 256;
    req["id"] = sid;
    req["apply_chat_template"] = true;
    req["max_generate_length_override"] = capOverride;

    Json userContent = Json::array();
    if (!st.melPath.empty())
    {
        userContent.push_back(Json{{"type", "audio"}, {"audio", st.melPath}});
    }
    Json messages = Json::array();
    messages.push_back(Json{{"role", "user"}, {"content", userContent}});
    if (prefix.empty())
    {
        req["add_generation_prompt"] = true;
    }
    else
    {
        req["add_generation_prompt"] = false;
        messages.push_back(Json{{"role", "assistant"}, {"content", prefix}});
    }
    req["requests"] = Json::array({Json{{"messages", messages}, {"reference", "ref"}}});
    return req;
}

//! Run a single chunk-and-confirm hop and return {ok, generated, total_ms}. The
//! generated text is the CONTINUATION only (lang preamble stripped via parseAsrText
//! for the warmup-hop case); the caller forms rawDecoded = prefix + generated.
HopOut runPrefixHop(std::string const& sid, SessionState const& st, std::string const& prefix, int32_t capOverride,
    bool isFinish, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    (void) isFinish;
    HopOut out;
    // capOverride: >0 caps the hop (non-final), <=0 (==-1) leaves it uncapped (final).
    Json const req = buildPrefixRequest(sid, st, prefix, capOverride);
    // Reuse the PROVEN parse->handleRequest core. runOneShotCore honors
    // apply_chat_template / add_generation_prompt / max_generate_length_override
    // straight from the request JSON. (We are inside OVS_ASR_STREAM_PREFIX, so
    // asrStreamDelta/asrSessionPath are not the active branch in runOneShotCore.)
    Json core = runOneShotCore(req, runtime, stream, loraWeightsMap);
    out.ok = core.value("ok", false);
    if (core.contains("total_ms")) out.totalMs = core["total_ms"].get<double>();
    if (out.ok && core.contains("responses") && core["responses"].is_array() && !core["responses"].empty())
    {
        Json const& r0 = core["responses"][0];
        if (r0.contains("output_text") && r0["output_text"].is_string())
            out.generated = parseAsrText(r0["output_text"].get<std::string>());
    }
    if (!out.ok && core.contains("error") && core["error"].is_string())
        out.error = core["error"].get<std::string>();
    return out;
}
} // namespace asrStreamPrefix

//! Core one-shot engine step: parse the `requests` payload, run handleRequest on
//! each, and return the assembled response Json. This is the PROVEN golden path
//! (CER 0.0000 + byte-identity). Both the legacy event-less one-shot path AND the
//! streaming `end`/finalize path call this, so finalize inherits exact parity.
//! NOT internally synchronized — callers must hold gEngineExecMutex (shared
//! PipelineIO is mutated by handleRequest).
//!
//! Phase 2a: when OVS_ASR_SESSION_PATH=1 this delegates to the session single-hop
//! path (asrSessionPath::runSessionCore) instead — an OFFLINE A/B equivalence
//! surface. Default (flag OFF) is unchanged: handleRequest, byte-identical to before.
Json runOneShotCore(Json input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    if (asrStreamDelta::enabled())
    {
        return asrStreamDelta::runDeltaCore(std::move(input), runtime, stream);
    }
    if (asrSessionPath::enabled())
    {
        return asrSessionPath::runSessionCore(std::move(input), runtime, stream);
    }
    Json response;
    std::filesystem::path tempPath;
    auto const requestStart = std::chrono::steady_clock::now();
    try
    {
        std::string const id = input.value("id", "");
        input.erase("id");
        int32_t const batchSizeOverride = input.value("batch_size_override", -1);
        int64_t const maxGenerateLengthOverride = input.value("max_generate_length_override", -1);
        input.erase("batch_size_override");
        input.erase("max_generate_length_override");

        tempPath = writeTempInput(input, id);
        std::vector<rt::LLMGenerationRequest> batchedRequests;
        std::tie(loraWeightsMap, batchedRequests)
            = exampleUtils::parseRequestFile(tempPath, batchSizeOverride, maxGenerateLengthOverride);
        if (batchedRequests.empty())
        {
            throw std::runtime_error("No valid ASR requests found");
        }

        Json responses = Json::array();
        bool ok = true;
        for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
        {
            rt::LLMGenerationResponse llmResponse;
            bool const requestOk = runtime.handleRequest(batchedRequests[requestIdx], llmResponse, stream);
            ok = ok && requestOk;
            for (size_t batchIdx = 0; batchIdx < batchedRequests[requestIdx].requests.size(); ++batchIdx)
            {
                bool const hasOutputText = requestOk && batchIdx < llmResponse.outputTexts.size();
                std::string const text = hasOutputText
                    ? llmResponse.outputTexts[batchIdx]
                    : "TensorRT Edge LLM cannot handle this request. Fails.";
                responses.push_back(Json{
                    {"request_idx", requestIdx}, {"batch_idx", batchIdx}, {"output_text", text}});
            }
        }
        double const totalMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - requestStart).count();
        if (!ok)
        {
            response = Json{{"id", id}, {"event", "error"}, {"ok", false}, {"responses", responses},
                {"total_ms", totalMs}};
        }
        else
        {
            response = Json{{"id", id}, {"event", "done"}, {"ok", true}, {"responses", responses},
                {"total_ms", totalMs}};
        }
    }
    catch (std::exception const& e)
    {
        response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}};
    }
    if (!tempPath.empty())
    {
        std::error_code ec;
        std::filesystem::remove(tempPath, ec);
    }
    return response;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    auto pluginHandles = loadEdgellmPluginLib();

    // Load mel-preprocessing assets if provided (restored from the v0.7.x worker;
    // the #9 refactor dropped this). PCM input via `pcm_b64` chunk events requires
    // both files; mel_path-only callers remain fully supported when these are
    // absent. The production stream_mode=worker path ALWAYS sends pcm_b64, so these
    // must be wired (the v080 profile sets EDGE_LLM_ASR_MEL_{SETTINGS,FILTERS}).
    if (!args.melSettingsPath.empty() && !args.melFiltersPath.empty())
    {
        try
        {
            gMelExtractor = std::make_unique<MelExtractor>(args.melSettingsPath, args.melFiltersPath);
            LOG_INFO("MelExtractor loaded (n_fft=%d n_mels=%d hop=%d) — pcm_b64 input enabled",
                gMelExtractor->n_fft(), gMelExtractor->n_mels(), gMelExtractor->hop_length());
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("MelExtractor init failed: %s — PCM input disabled", e.what());
            gMelExtractor.reset();
        }
    }

    setProfilingEnabled(true);

    auto const initStart = std::chrono::steady_clock::now();
    std::unordered_map<std::string, std::string> loraWeightsMap;
    bool const enableGraph = std::getenv("EDGE_LLM_ASR_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_ASR_CUDA_GRAPH")) != "0";
    cudaStream_t stream{nullptr};
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto runtimePtr = std::make_unique<rt::LLMInferenceRuntime>(
        args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    if (enableGraph && !runtimePtr->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for ASR one-shot runtime, proceeding without.");
    }
    rt::LLMInferenceRuntime* runtime = runtimePtr.get();

    // Streaming-prefix rollback (OVS_ASR_STREAM_PREFIX=1): load the SAME tokenizer
    // the engine uses (engineDir/tokenizer.json + processed_chat_template.json) so
    // computePrefix can encode/decode the accumulated transcript for the
    // unfixedTokenNum roll-back. Loaded ONLY when the flag is ON; the runtime decode
    // path is untouched either way. A failed load leaves gPrefixTokenizer null and
    // computePrefix degrades to "no prefix" (cumulative-equivalent, never crashes).
    if (asrStreamPrefix::enabled())
    {
        auto tk = std::make_unique<tokenizer::Tokenizer>();
        if (tk->loadFromHF(args.engineDir))
        {
            std::filesystem::path const ctPath
                = std::filesystem::path(args.engineDir) / "processed_chat_template.json";
            if (std::filesystem::exists(ctPath)) tk->loadChatTemplate(ctPath);
            // Round-trip sanity check (stderr; does not perturb the stdout protocol).
            std::string const probe = "language Chineseä½ å¥½ä¸ç";
            auto ids = tk->encode(probe, false, false);
            std::string rt2 = tk->decode(ids, false);
            std::cerr << "[prefix-tok] loaded numVocab=" << tk->getNumVocab() << " probe_tokens=" << ids.size()
                      << " roundtrip_ok=" << (rt2 == probe ? 1 : 0) << " decoded=" << rt2 << std::endl;
            gPrefixTokenizer = std::move(tk);
        }
        else
        {
            std::cerr << "[prefix-tok] loadFromHF FAILED for " << args.engineDir
                      << " -> prefix rollback degrades to no-prefix" << std::endl;
        }
    }

    // Task #9: lane reservation pool. asrMax = requested --max_slots, clamped to
    // the engine's physical batch capacity (maxSessionBatchSize). The b2 engine
    // exposes 2 rows; the b1 engine exposes 1 (so N>1 silently degrades to N=1
    // there). TTS partition is 0 in this ASR-only worker.
    int32_t const physBatch = runtime->maxSessionBatchSize();
    int32_t asrMax = args.maxSlots;
    if (asrMax > physBatch) asrMax = physBatch;
    if (asrMax < 1) asrMax = 1;
    gMaxSlots = asrMax;
    gLaneMgr = std::make_unique<rt::SessionLaneManager>(physBatch, asrMax, /*ttsMax=*/0);

    double const initMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}, {"max_slots", gMaxSlots}}.dump() << std::endl;

    // Streaming worker stdin loop. Use poll() with a 1 s timeout so we can fire
    // the mandatory idle-timeout sweep between events (half-open lane-leak guard).
    auto const checkIdleTimeout = []() { sweepIdleSessions(); };

    std::string buffer;
    char readBuf[4096];
    bool eof = false;
    while (!eof)
    {
        struct pollfd pfd = {STDIN_FILENO, POLLIN, 0};
        int const pr = ::poll(&pfd, 1, 1000);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            LOG_ERROR("poll() failed: %s", std::strerror(errno));
            break;
        }
        if (pr == 0)
        {
            checkIdleTimeout();
            continue;
        }
        if (pfd.revents & (POLLIN | POLLHUP))
        {
            auto const n = ::read(STDIN_FILENO, readBuf, sizeof(readBuf));
            if (n <= 0)
            {
                eof = true;
                break;
            }
            buffer.append(readBuf, static_cast<size_t>(n));
        }
        // Process whole lines accumulated in the buffer.
        size_t pos;
        while ((pos = buffer.find('\n')) != std::string::npos)
        {
            std::string const line = buffer.substr(0, pos);
            buffer.erase(0, pos + 1);
            if (line.empty())
            {
                continue;
            }

            Json parsed;
            try
            {
                parsed = Json::parse(line);
            }
            catch (std::exception const& e)
            {
                Json err = {{"event", "error"}, {"ok", false},
                    {"error", std::string("json_parse_failed: ") + e.what()}};
                std::cout << err.dump() << std::endl;
                // Step 3 — malformed input carries no parseable sessionId, so we
                // cannot target a specific slot to drop. The idle-timeout sweep
                // reaps any genuinely stuck slot after kIdleTimeoutMs; emitting
                // the error and continuing is sufficient.
                continue;
            }

            if (!parsed.contains("event"))
            {
                // Backward-compat one-shot: any line that omits `event` flows
                // through the legacy path. BYTE-IDENTICAL with v080-0016: same
                // core, same response shape. Mutex held for forward-compat (no
                // contention single-threaded) so the engine step is identical.
                Json response;
                {
                    std::lock_guard<std::mutex> lk(gEngineExecMutex);
                    response = runOneShotCore(std::move(parsed), *runtime, stream, loraWeightsMap);
                }
                std::cout << response.dump() << std::endl;
                continue;
            }

            std::string const event = parsed.value("event", "");
            if (event == "begin")
            {
                handleBegin(parsed);
            }
            else if (event == "chunk")
            {
                handleChunk(parsed, *runtime, stream, loraWeightsMap);
            }
            else if (event == "end")
            {
                handleEnd(parsed, *runtime, stream, loraWeightsMap);
            }
            else
            {
                Json err = {{"event", "error"}, {"ok", false}, {"error", "unknown_event"},
                    {"received", event}};
                std::string const evId = parsed.value("id", "");
                if (!evId.empty())
                {
                    err["id"] = evId;
                }
                std::cout << err.dump() << std::endl;
            }
        }
        checkIdleTimeout();
    }

    runtimePtr.reset();
    if (stream != nullptr)
    {
        cudaStreamDestroy(stream);
    }
    return EXIT_SUCCESS;
}
