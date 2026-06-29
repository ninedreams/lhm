#include <thread>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cinttypes>
#include <climits>

#include "config.h"

// ============================================================================
// Log parameters (originally in config.h)
// ============================================================================
DEFINE_string(log_file, "logs/lhm.log", "Log file path");
DEFINE_int32(log_rotate_hour, 3, "Log rotate hour");
DEFINE_int32(log_rotate_minute, 0, "Log rotate minute");
DEFINE_string(log_level, "info", "Log level (trace, debug, info, warn, error, critical, off), if not found set off.");
DEFINE_string(log_pattern, "[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v", "Log pattern (see spdlog documentation for details)");

// ============================================================================
// General parameters
// ============================================================================
DEFINE_bool(verbose_prompt, false, "print a verbose prompt before generation");
DEFINE_bool(display_prompt, true, "whether to print prompt at generation");
DEFINE_string(color, "auto", "Colorize output ('on', 'off', or 'auto')");
DEFINE_int32(threads, 0, "number of CPU threads to use during generation (0 = auto)");
DEFINE_int32(threads_batch, 0, "number of threads to use during batch and prompt processing (0 = same as --threads)");
DEFINE_string(cpu_mask, "", "CPU affinity mask: arbitrarily long hex. Complements cpu-range");
DEFINE_string(cpu_range, "", "range of CPUs for affinity. Complements --cpu-mask");
DEFINE_string(cpu_strict, "0", "use strict CPU placement (0 or 1)");
DEFINE_int32(prio, 0, "set process/thread priority: low(-1), normal(0), medium(1), high(2), realtime(3)");
DEFINE_string(poll, "0", "use polling level to wait for work (0 = no polling)");
DEFINE_string(cpu_mask_batch, "", "CPU affinity mask for batch. Complements cpu-range-batch");
DEFINE_string(cpu_range_batch, "", "ranges of CPUs for affinity for batch. Complements --cpu-mask-batch");
DEFINE_int32(cpu_strict_batch, -1, "use strict CPU placement for batch (-1 = same as --cpu-strict)");
DEFINE_int32(prio_batch, -1, "set process/thread priority for batch (-1 = same as --prio)");
DEFINE_int32(poll_batch, -1, "use polling to wait for work for batch (-1 = same as --poll)");

// ============================================================================
// Context / batch parameters
// ============================================================================
DEFINE_int32(ctx_size, 0, "size of the prompt context (0 = loaded from model)");
DEFINE_int32(predict, -1, "number of tokens to predict (-1 = infinity, -2 = until context filled)");
DEFINE_int32(batch_size, 2048, "logical maximum batch size");
DEFINE_int32(ubatch_size, 512, "physical maximum batch size");
DEFINE_int32(keep, 0, "number of tokens to keep from the initial prompt (-1 = all)");
DEFINE_bool(swa_full, false, "use full-size SWA cache");
DEFINE_int32(ctx_checkpoints, 32, "max number of context checkpoints to create per slot");
DEFINE_int32(checkpoint_min_step, 256, "minimum spacing between context checkpoints in tokens");
DEFINE_int32(cache_ram, 8192, "set the maximum cache size in MiB (-1 = no limit, 0 = disable)");
DEFINE_bool(kv_unified, false, "use single unified KV buffer shared across all sequences");
DEFINE_bool(cache_idle_slots, true, "save idle slots to the prompt cache on new task");
DEFINE_bool(context_shift, false, "whether to use context shift on infinite text generation");
DEFINE_int32(chunks, -1, "max number of chunks to process (-1 = all)");

// ============================================================================
// Flash attention
// ============================================================================
DEFINE_string(flash_attn, "auto", "set Flash Attention use ('on', 'off', or 'auto')");

// ============================================================================
// Prompt parameters
// ============================================================================
DEFINE_string(prompt, "", "prompt to start generation with");
DEFINE_string(system_prompt, "", "system prompt to use with model");
DEFINE_bool(perf, true, "whether to enable internal libllama performance timings");
DEFINE_bool(show_timings, true, "whether to show timing information after each response");
DEFINE_string(file, "", "a file containing the prompt");
DEFINE_string(system_prompt_file, "", "a file containing the system prompt");
DEFINE_string(in_file, "", "an input file (use comma-separated values to specify multiple files)");
DEFINE_string(binary_file, "", "binary file containing the prompt");
DEFINE_bool(escape, true, "whether to process escapes sequences (\\n, \\r, \\t, etc.)");
DEFINE_int32(print_token_count, -1, "print token count every N tokens (-1 = disabled)");
DEFINE_string(prompt_cache, "", "path to file for saving/loading prompt eval state");
DEFINE_bool(prompt_cache_all, false, "save user input and generations to prompt cache");
DEFINE_bool(prompt_cache_ro, false, "open the prompt cache read-only");
DEFINE_string(reverse_prompt, "", "string upon which more user input is prompted (can be repeated)");
DEFINE_bool(special, false, "enable special token output");
DEFINE_bool(conversation, false, "enable conversation mode");
DEFINE_bool(single_turn, false, "single turn chat conversation");
DEFINE_bool(interactive, false, "run in interactive mode");
DEFINE_bool(interactive_first, false, "run in interactive mode and wait for input right away");
DEFINE_bool(multiline_input, false, "allows you to write or paste multiple lines without ending each in '\\'");
DEFINE_bool(in_prefix_bos, false, "prefix BOS to user inputs, preceding the --in-prefix string");
DEFINE_string(in_prefix, "", "string to prefix user inputs with");
DEFINE_string(in_suffix, "", "string to suffix after user inputs with");
DEFINE_bool(warmup, true, "whether to perform warmup with an empty run");
DEFINE_bool(spm_infill, false, "use Suffix/Prefix/Middle pattern for infill");

// ============================================================================
// Sampling parameters
// ============================================================================
DEFINE_string(samplers, "", "samplers that will be used for generation, separated by ';'");
DEFINE_string(seed, "", "RNG seed (use random seed for -1)");
DEFINE_string(sampler_seq, "", "simplified sequence for samplers");
DEFINE_bool(ignore_eos, false, "ignore end of stream token and continue generating");
DEFINE_string(temp, "", "temperature");
DEFINE_int32(top_k, 40, "top-k sampling (0 = disabled)");
DEFINE_string(top_p, "", "top-p sampling (1.0 = disabled)");
DEFINE_string(min_p, "", "min-p sampling (0.0 = disabled)");
DEFINE_string(top_nsigma, "", "top-n-sigma sampling (-1.0 = disabled)");
DEFINE_string(xtc_probability, "", "xtc probability (0.0 = disabled)");
DEFINE_string(xtc_threshold, "", "xtc threshold (1.0 = disabled)");
DEFINE_string(typical, "", "locally typical sampling, parameter p (1.0 = disabled)");
DEFINE_int32(repeat_last_n, 64, "last n tokens to consider for penalize (0 = disabled, -1 = ctx_size)");
DEFINE_string(repeat_penalty, "", "penalize repeat sequence of tokens (1.0 = disabled)");
DEFINE_string(presence_penalty, "", "repeat alpha presence penalty (0.0 = disabled)");
DEFINE_string(frequency_penalty, "", "repeat alpha frequency penalty (0.0 = disabled)");
DEFINE_string(dry_multiplier, "", "set DRY sampling multiplier (0.0 = disabled)");
DEFINE_string(dry_base, "", "set DRY sampling base value");
DEFINE_int32(dry_allowed_length, 2, "set allowed length for DRY sampling");
DEFINE_int32(dry_penalty_last_n, -1, "set DRY penalty for the last n tokens (-1 = context size)");
DEFINE_string(dry_sequence_breaker, "", "add sequence breaker for DRY sampling");
DEFINE_string(adaptive_target, "", "adaptive sampling target");
DEFINE_string(adaptive_decay, "", "adaptive sampling decay");
DEFINE_string(dynatemp_range, "", "dynatemp range");
DEFINE_string(dynatemp_exp, "", "dynatemp exponent");
DEFINE_int32(mirostat, 0, "use Mirostat sampling (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)");
DEFINE_string(mirostat_lr, "", "Mirostat learning rate eta");
DEFINE_string(mirostat_ent, "", "Mirostat target entropy tau");
DEFINE_string(logit_bias, "", "logit bias for specific tokens (TOKEN_ID(+/-)BIAS)");
DEFINE_string(grammar, "", "BNF-like grammar to constrain generations");
DEFINE_string(grammar_file, "", "file to read grammar from");
DEFINE_string(json_schema, "", "JSON schema to constrain generations");
DEFINE_string(json_schema_file, "", "file to read JSON schema from");
DEFINE_bool(backend_sampling, false, "use backend sampling");

// ============================================================================
// Model type parameters
// ============================================================================
DEFINE_string(pooling, "", "pooling type for embeddings");
DEFINE_string(attention, "", "attention type for embeddings");
DEFINE_string(rope_scaling, "", "RoPE scaling type");
DEFINE_string(rope_scale, "", "RoPE frequency scaling factor");
DEFINE_string(rope_freq_base, "", "RoPE base frequency");
DEFINE_string(rope_freq_scale, "", "RoPE frequency scaling factor");
DEFINE_int32(yarn_orig_ctx, 0, "YaRN original context length");
DEFINE_string(yarn_ext_factor, "", "YaRN extrapolation mix factor");
DEFINE_string(yarn_attn_factor, "", "YaRN magnitude scaling factor");
DEFINE_string(yarn_beta_slow, "", "YaRN low correction dim");
DEFINE_string(yarn_beta_fast, "", "YaRN high correction dim");
DEFINE_int32(grp_attn_n, 1, "group-attention factor");
DEFINE_int32(grp_attn_w, 512, "group-attention width");

// ============================================================================
// KV cache / offload parameters
// ============================================================================
DEFINE_bool(kv_offload, true, "enable KV offloading to GPU");
DEFINE_bool(repack, true, "enable weight repacking (extra buffer types)");
DEFINE_bool(no_host, false, "bypass host buffer allowing extra buffers to be used");
DEFINE_string(cache_type_k, "f16", "KV cache data type for the K");
DEFINE_string(cache_type_v, "f16", "KV cache data type for the V");
DEFINE_string(defrag_thold, "", "KV cache defragmentation threshold");

// ============================================================================
// Parallel / batching
// ============================================================================
DEFINE_int32(parallel, 1, "number of parallel sequences to decode");
DEFINE_int32(sequences, 1, "number of sequences to decode");
DEFINE_bool(cont_batching, true, "insert new sequences for decoding on-the-fly");

// ============================================================================
// RPC / memory parameters
// ============================================================================
DEFINE_string(rpc, "", "comma-separated list of RPC servers");
DEFINE_bool(mlock, false, "use mlock to keep model in memory");
DEFINE_bool(mmap, true, "enable mmap to use filesystem cache");
DEFINE_bool(direct_io, false, "read from disk without buffering");
DEFINE_string(numa, "", "NUMA optimization strategy");
DEFINE_string(device, "", "comma-separated list of devices to use for offloading");
DEFINE_bool(list_devices, false, "list available devices");
DEFINE_string(override_tensor, "", "override tensor buffer type (tensor_name=buffer_type)");
DEFINE_bool(cpu_moe, false, "offload MoE layers to CPU");
DEFINE_int32(n_cpu_moe, 0, "number of MoE layers to offload to CPU");

// ============================================================================
// GPU parameters
// ============================================================================
DEFINE_string(gpu_layers, "-1", "number of layers to store in VRAM (-1 = auto, <= -2 = all)");
DEFINE_string(split_mode, "", "how to split the model across GPUs");
DEFINE_string(tensor_split, "", "how split tensors should be distributed across GPUs");
DEFINE_int32(main_gpu, 0, "the GPU that is used for scratch and small tensors");

// ============================================================================
// Fit parameters
// ============================================================================
DEFINE_string(fit, "", "fit params (true/false)");
DEFINE_string(fit_print, "", "print fit params (true/false)");
DEFINE_string(fit_target, "", "fit target per device in bytes");
DEFINE_int32(fit_ctx, 4096, "minimum context size to set when trying to reduce memory use");
DEFINE_bool(check_tensors, false, "validate tensor data");

// ============================================================================
// Override / LoRA / control vector parameters
// ============================================================================
DEFINE_string(override_kv, "", "override model metadata key-value pairs");
DEFINE_bool(op_offload, false, "globally disable offload host tensor operations to device");
DEFINE_string(lora, "", "path to LoRA adapter (can be repeated)");
DEFINE_string(lora_scaled, "", "path to LoRA adapter with scale (PATH SCALE)");
DEFINE_string(control_vector, "", "path to control vector (can be repeated)");
DEFINE_string(control_vector_scaled, "", "path to control vector with scale (PATH SCALE)");
DEFINE_string(control_vector_layer_range, "", "layer range for control vector (START END)");
DEFINE_bool(lora_init_without_apply, false, "only load lora to memory, but do not apply it");

// ============================================================================
// Model path parameters
// ============================================================================
DEFINE_string(alias, "", "model alias(es)");
DEFINE_string(tags, "", "model tags");
DEFINE_string(model, "", "model path");

// ============================================================================
// Retrieval / embedding parameters
// ============================================================================
DEFINE_string(context_file, "", "context file(s) to embed (comma-separated)");
DEFINE_int32(chunk_size, 64, "chunk size for context embedding");
DEFINE_string(chunk_separator, "\n", "chunk separator for context embedding");
DEFINE_int32(embd_normalize, 2, "normalisation for embeddings (-1=none, 0=max, 1=taxicab, 2=euclidean)");
DEFINE_string(embd_output_format, "", "embedding output format");
DEFINE_string(embd_separator, "\n", "separator of embeddings");
DEFINE_string(cls_separator, "\t", "separator of classification sequences");
DEFINE_bool(embedding, false, "get only sentence embedding");
DEFINE_bool(rerank, false, "enable reranking");

// ============================================================================
// Passkey / benchmark parameters
// ============================================================================
DEFINE_int32(junk, 250, "number of times to repeat the junk text");
DEFINE_int32(pos, -1, "position of the passkey in the junk text");

// ============================================================================
// Imatrix parameters
// ============================================================================
DEFINE_string(output, "", "output filename");
DEFINE_int32(output_frequency, 10, "output the imatrix every N iterations");
DEFINE_string(output_format, "", "output format (gguf or dat)");
DEFINE_int32(save_frequency, 0, "save the imatrix every N iterations");
DEFINE_bool(process_output, false, "collect data for the output tensor");
DEFINE_bool(ppl, true, "whether to compute perplexity");
DEFINE_int32(chunk, 0, "start processing from this chunk");
DEFINE_bool(show_statistics, false, "show imatrix statistics per tensor");
DEFINE_bool(parse_special, false, "whether to parse special tokens during imatrix tokenization");

// ============================================================================
// Bench parameters
// ============================================================================
DEFINE_bool(pps, false, "is prompt processing shared");
DEFINE_bool(tgs, false, "is text generation separate");
DEFINE_string(npp, "", "prompt processing sizes (comma-separated)");
DEFINE_string(ntg, "", "text generation sizes (comma-separated)");
DEFINE_string(npl, "", "parallel sizes (comma-separated)");

// ============================================================================
// Perplexity parameters
// ============================================================================
DEFINE_int32(ppl_stride, 0, "stride for perplexity calculations");
DEFINE_int32(ppl_output_type, 0, "perplexity output type");
DEFINE_bool(hellaswag, false, "compute HellaSwag score");
DEFINE_int32(hellaswag_tasks, 400, "number of HellaSwag tasks");
DEFINE_bool(winogrande, false, "compute Winogrande score");
DEFINE_int32(winogrande_tasks, 0, "number of Winogrande tasks");
DEFINE_bool(multiple_choice, false, "compute TruthfulQA score");
DEFINE_int32(multiple_choice_tasks, 0, "number of TruthfulQA tasks");
DEFINE_bool(kl_divergence, false, "compute KL divergence");
DEFINE_string(save_all_logits, "", "file for saving all logits");

// ============================================================================
// Server parameters
// ============================================================================
DEFINE_string(host, "127.0.0.1", "hostname to bind the server to");
DEFINE_int32(port, 8080, "port to bind the server to");
DEFINE_bool(reuse_port, false, "allow multiple sockets to bind to the same port");
DEFINE_string(path, "", "path to serve static files from");
DEFINE_string(api_prefix, "", "API prefix for server endpoints");
DEFINE_string(tools, "", "enable built-in tools (comma-separated)");
DEFINE_string(api_key, "", "API key for server authentication");
DEFINE_string(api_key_file, "", "file containing API key");
DEFINE_string(ssl_key_file, "", "path to SSL key file");
DEFINE_string(ssl_cert_file, "", "path to SSL certificate file");
DEFINE_string(chat_template_kwargs, "", "additional kwargs for chat template (JSON)");
DEFINE_int32(timeout, 3600, "http read/write timeout in seconds");
DEFINE_int32(sse_ping_interval, 30, "SSE ping interval in seconds");
DEFINE_int32(threads_http, -1, "number of threads to process HTTP requests");
DEFINE_bool(cache_prompt, true, "whether to enable prompt caching");
DEFINE_int32(cache_reuse, 0, "min chunk size to reuse from the cache via KV shifting");
DEFINE_bool(metrics, false, "enable metrics endpoint");
DEFINE_bool(props, false, "enable props endpoint");
DEFINE_bool(slots, true, "enable slots endpoint");
DEFINE_string(slot_save_path, "", "path to save slot data");
DEFINE_bool(jinja, true, "use Jinja2 chat template");
DEFINE_string(reasoning_format, "", "reasoning format (deepseek, etc.)");
DEFINE_string(reasoning, "", "enable reasoning content (on/off/auto)");
DEFINE_int32(reasoning_budget, -1, "reasoning budget in tokens");
DEFINE_string(reasoning_budget_message, "", "reasoning budget message");
DEFINE_string(chat_template, "", "chat template string");
DEFINE_string(chat_template_file, "", "chat template file");
DEFINE_bool(skip_chat_parsing, false, "force pure content parser");
DEFINE_bool(prefill_assistant, true, "prefill trailing assistant message into the response");
DEFINE_string(slot_prompt_similarity, "", "slot prompt similarity threshold");
DEFINE_int32(sleep_idle_seconds, -1, "server will sleep after this many seconds of idle time");
DEFINE_bool(simple_io, false, "improves compatibility with subprocesses and limited consoles");

// ============================================================================
// Control vector generator parameters
// ============================================================================
DEFINE_string(positive_file, "tools/cvector-generator/positive.txt", "positive file for control vector");
DEFINE_string(negative_file, "tools/cvector-generator/negative.txt", "negative file for control vector");
DEFINE_int32(pca_batch, 100, "PCA batch size");
DEFINE_int32(pca_iter, 1000, "PCA iterations");
DEFINE_string(method, "", "dimensionality reduction method");

// ============================================================================
// Logging parameters (server-specific)
// ============================================================================
DEFINE_bool(log_disable, false, "disable logging");
DEFINE_bool(log_file_flag, false, "log to file flag");
DEFINE_string(log_prompts_dir, "", "directory with logged prompts");
DEFINE_bool(log_colors, true, "use colors in log output");
DEFINE_bool(verbose, false, "verbose output");
DEFINE_bool(offline, false, "offline mode");
DEFINE_int32(verbosity, 3, "log verbosity level");
DEFINE_bool(log_prefix, true, "use prefix in log output");
DEFINE_bool(log_timestamps, true, "use timestamps in log output");

// ============================================================================
// Speculative decoding parameters
// ============================================================================
DEFINE_string(spec_draft_hf, "", "Hugging Face repo for draft model");
DEFINE_int32(spec_draft_threads, 0, "number of CPU threads for draft model");
DEFINE_int32(spec_draft_threads_batch, 0, "number of threads for draft model batch processing");
DEFINE_string(spec_draft_cpu_mask, "", "CPU affinity mask for draft model");
DEFINE_string(spec_draft_cpu_range, "", "CPU range for draft model");
DEFINE_int32(spec_draft_cpu_strict, -1, "strict CPU placement for draft model");
DEFINE_int32(spec_draft_prio, -1, "process/thread priority for draft model");
DEFINE_int32(spec_draft_poll, -1, "polling level for draft model");
DEFINE_string(spec_draft_cpu_mask_batch, "", "CPU affinity mask for draft model batch");
DEFINE_bool(spec_draft_cpu_range_batch, false, "CPU range for draft model batch");
DEFINE_int32(spec_draft_cpu_strict_batch, -1, "strict CPU placement for draft model batch");
DEFINE_int32(spec_draft_prio_batch, -1, "process/thread priority for draft model batch");
DEFINE_int32(spec_draft_poll_batch, -1, "polling level for draft model batch");
DEFINE_string(spec_draft_type_k, "", "KV cache data type for K in draft model");
DEFINE_string(spec_draft_type_v, "", "KV cache data type for V in draft model");
DEFINE_string(spec_draft_override_tensor, "", "override tensor buffer type for draft model");
DEFINE_bool(spec_draft_cpu_moe, false, "offload MoE layers to CPU for draft model");
DEFINE_int32(spec_draft_n_cpu_moe, 0, "number of MoE layers to offload to CPU for draft model");
DEFINE_int32(spec_draft_n_max, 0, "max draft tokens");
DEFINE_int32(spec_draft_n_min, 0, "min draft tokens");
DEFINE_string(spec_draft_p_split, "", "draft p-split value");
DEFINE_string(spec_draft_p_min, "", "draft p-min value");
DEFINE_bool(spec_draft_backend_sampling, false, "use backend sampling for draft model");
DEFINE_string(spec_draft_device, "", "device for draft model");
DEFINE_string(spec_draft_ngl, "", "number of GPU layers for draft model");
DEFINE_string(spec_draft_model, "", "path to draft model");
DEFINE_string(spec_type, "", "speculative decoding type(s)");
DEFINE_int32(spec_ngram_mod_n_min, 0, "ngram mod n_min");
DEFINE_int32(spec_ngram_mod_n_max, 0, "ngram mod n_max");
DEFINE_int32(spec_ngram_mod_n_match, 0, "ngram mod n_match");
DEFINE_int32(spec_ngram_simple_size_n, 0, "ngram simple size_n");
DEFINE_int32(spec_ngram_simple_size_m, 0, "ngram simple size_m");
DEFINE_int32(spec_ngram_simple_min_hits, 0, "ngram simple min_hits");
DEFINE_int32(spec_ngram_map_k_size_n, 0, "ngram map_k size_n");
DEFINE_int32(spec_ngram_map_k_size_m, 0, "ngram map_k size_m");
DEFINE_int32(spec_ngram_map_k_min_hits, 0, "ngram map_k min_hits");
DEFINE_int32(spec_ngram_map_k4v_size_n, 0, "ngram map_k4v size_n");
DEFINE_int32(spec_ngram_map_k4v_size_m, 0, "ngram map_k4v size_m");
DEFINE_int32(spec_ngram_map_k4v_min_hits, 0, "ngram map_k4v min_hits");
DEFINE_bool(draft, false, "enable draft speculative decoding");
DEFINE_bool(draft_min, false, "enable draft min speculative decoding");
DEFINE_bool(spec_ngram_size_n, false, "ngram size_n (deprecated)");
DEFINE_bool(spec_ngram_size_m, false, "ngram size_m (deprecated)");
DEFINE_bool(spec_ngram_min_hits, false, "ngram min_hits (deprecated)");

// ============================================================================
// Lookup cache parameters
// ============================================================================
DEFINE_string(lookup_cache_static, "", "path to static lookup cache");
DEFINE_string(lookup_cache_dynamic, "", "path to dynamic lookup cache");

// ============================================================================
// Vocoder parameters
// ============================================================================
DEFINE_string(model_vocoder, "", "path to vocoder model");
DEFINE_bool(tts_use_guide_tokens, false, "use guide tokens for TTS");
DEFINE_string(tts_speaker_file, "", "path to TTS speaker file");
DEFINE_bool(tts_oute_default, false, "use default OUTE TTS configuration");

// ============================================================================
// Diffusion parameters
// ============================================================================
DEFINE_int32(diffusion_steps, 0, "number of diffusion steps");
DEFINE_bool(diffusion_visual, false, "enable visual mode for diffusion");
DEFINE_string(diffusion_eps, "", "diffusion epsilon");
DEFINE_int32(diffusion_algorithm, 0, "diffusion algorithm");
DEFINE_string(diffusion_alg_temp, "", "diffusion algorithm temperature");
DEFINE_int32(diffusion_block_length, 0, "diffusion block length");
DEFINE_string(diffusion_cfg_scale, "", "diffusion CFG scale");
DEFINE_string(diffusion_add_gumbel_noise, "", "add Gumbel noise for diffusion");

// ============================================================================
// Finetune parameters
// ============================================================================
DEFINE_string(learning_rate, "", "learning rate");
DEFINE_string(learning_rate_min, "", "minimum learning rate");
DEFINE_string(learning_rate_decay_epochs, "", "learning rate decay epochs");
DEFINE_string(weight_decay, "", "weight decay");
DEFINE_string(val_split, "", "validation split fraction");
DEFINE_int32(epochs, 0, "number of training epochs");
DEFINE_string(optimizer, "", "optimizer type");

// ============================================================================
// Debug parameters
// ============================================================================
DEFINE_bool(check, false, "check rather than generate results");
DEFINE_bool(save_logits, false, "whether to save logits to files");
DEFINE_string(logits_output_dir, "data", "directory for saving logits output files");
DEFINE_string(tensor_filter, "", "filter tensor names for debug output (regex)");

// ============================================================================
// Preset defaults
// ============================================================================
DEFINE_bool(embd_gemma_default, false, "use default Gemma embedding configuration");
DEFINE_bool(spec_default, false, "use default speculative decoding configuration");


namespace lhm {

void init_config(int argc, char ** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

} // namespace lhm
