#include "config.h"
#include "common.h"
#include "log.h"
#include "download.h"
#include "speculative.h"

#include <thread>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cinttypes>
#include <climits>

// Helper: read file contents
static std::string read_file_contents(const std::string & fname) {
    std::ifstream file(fname);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

// Helper: parse CSV row
static std::vector<std::string> parse_csv_row(const std::string & value) {
    return string_split<std::string>(value, ',');
}

// Helper: check if a string flag was explicitly set (non-empty for string, non-default for others)
// For string flags, empty means not set
static bool is_set(const std::string & val) {
    return !val.empty();
}

// Helper: parse bool-like string
static bool parse_bool_string(const std::string & val) {
    if (val == "1" || val == "true" || val == "yes" || val == "on") return true;
    if (val == "0" || val == "false" || val == "no" || val == "off") return false;
    throw std::invalid_argument(string_format("invalid boolean value: '%s'", val.c_str()));
}

namespace lhm {

void init_config(int argc, char ** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

void fill_common_params(common_params & params) {
    // ========================================================================
    // General parameters
    // ========================================================================
    if (FLAGS_verbose_prompt)    params.verbose_prompt = true;
    if (!FLAGS_display_prompt)   params.display_prompt = false;

    // --color
    if (is_set(FLAGS_color)) {
        if (FLAGS_color == "on" || FLAGS_color == "1" || FLAGS_color == "true") {
            params.use_color = true;
        } else if (FLAGS_color == "off" || FLAGS_color == "0" || FLAGS_color == "false") {
            params.use_color = false;
        } else if (FLAGS_color == "auto") {
            params.use_color = tty_can_use_colors();
        }
    }

    // --threads
    if (FLAGS_threads != 0) {
        params.cpuparams.n_threads = FLAGS_threads;
        if (params.cpuparams.n_threads <= 0) {
            params.cpuparams.n_threads = std::thread::hardware_concurrency();
        }
    }

    // --threads-batch
    if (FLAGS_threads_batch != 0) {
        params.cpuparams_batch.n_threads = FLAGS_threads_batch;
        if (params.cpuparams_batch.n_threads <= 0) {
            params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
        }
    }

    // --cpu-mask
    if (is_set(FLAGS_cpu_mask)) {
        params.cpuparams.mask_valid = true;
        if (!parse_cpu_mask(FLAGS_cpu_mask, params.cpuparams.cpumask)) {
            throw std::invalid_argument("invalid cpumask");
        }
    }

    // --cpu-range
    if (is_set(FLAGS_cpu_range)) {
        params.cpuparams.mask_valid = true;
        if (!parse_cpu_range(FLAGS_cpu_range, params.cpuparams.cpumask)) {
            throw std::invalid_argument("invalid range");
        }
    }

    // --cpu-strict
    if (is_set(FLAGS_cpu_strict)) {
        params.cpuparams.strict_cpu = std::stoul(FLAGS_cpu_strict);
    }

    // --prio
    if (FLAGS_prio != 0) {
        if (FLAGS_prio < GGML_SCHED_PRIO_LOW || FLAGS_prio > GGML_SCHED_PRIO_REALTIME) {
            throw std::invalid_argument("invalid value for --prio");
        }
        params.cpuparams.priority = (enum ggml_sched_priority) FLAGS_prio;
    }

    // --poll
    if (is_set(FLAGS_poll)) {
        params.cpuparams.poll = std::stoul(FLAGS_poll);
    }

    // --cpu-mask-batch
    if (is_set(FLAGS_cpu_mask_batch)) {
        params.cpuparams_batch.mask_valid = true;
        if (!parse_cpu_mask(FLAGS_cpu_mask_batch, params.cpuparams_batch.cpumask)) {
            throw std::invalid_argument("invalid cpumask");
        }
    }

    // --cpu-range-batch
    if (is_set(FLAGS_cpu_range_batch)) {
        params.cpuparams_batch.mask_valid = true;
        if (!parse_cpu_range(FLAGS_cpu_range_batch, params.cpuparams_batch.cpumask)) {
            throw std::invalid_argument("invalid range");
        }
    }

    // --cpu-strict-batch
    if (FLAGS_cpu_strict_batch >= 0) {
        params.cpuparams_batch.strict_cpu = FLAGS_cpu_strict_batch;
    }

    // --prio-batch
    if (FLAGS_prio_batch >= 0) {
        if (FLAGS_prio_batch < 0 || FLAGS_prio_batch > 3) {
            throw std::invalid_argument("invalid value for --prio-batch");
        }
        params.cpuparams_batch.priority = (enum ggml_sched_priority) FLAGS_prio_batch;
    }

    // --poll-batch
    if (FLAGS_poll_batch >= 0) {
        params.cpuparams_batch.poll = FLAGS_poll_batch;
    }

    // ========================================================================
    // Context / batch parameters
    // ========================================================================
    if (FLAGS_ctx_size != 0) {
        params.n_ctx = FLAGS_ctx_size;
        if (FLAGS_ctx_size == 0) {
            params.fit_params_min_ctx = UINT32_MAX;
        }
    }

    if (FLAGS_predict != -1) {
        params.n_predict = FLAGS_predict;
    }

    if (FLAGS_batch_size != 2048) {
        params.n_batch = FLAGS_batch_size;
    }

    if (FLAGS_ubatch_size != 512) {
        params.n_ubatch = FLAGS_ubatch_size;
    }

    if (FLAGS_keep != 0) {
        params.n_keep = FLAGS_keep;
    }

    if (FLAGS_swa_full) {
        params.swa_full = true;
    }

    if (FLAGS_ctx_checkpoints != 32) {
        params.n_ctx_checkpoints = FLAGS_ctx_checkpoints;
    }

    if (FLAGS_checkpoint_min_step != 256) {
        if (FLAGS_checkpoint_min_step < 0) {
            throw std::invalid_argument("checkpoint-min-step must be non-negative");
        }
        params.checkpoint_min_step = FLAGS_checkpoint_min_step;
    }

    if (FLAGS_cache_ram != 8192) {
        params.cache_ram_mib = FLAGS_cache_ram;
    }

    if (FLAGS_kv_unified) {
        params.kv_unified = true;
    }

    if (!FLAGS_cache_idle_slots) {
        params.cache_idle_slots = false;
    }

    if (FLAGS_context_shift) {
        params.ctx_shift = true;
    }

    if (FLAGS_chunks != -1) {
        params.n_chunks = FLAGS_chunks;
    }

    // ========================================================================
    // Flash attention
    // ========================================================================
    if (is_set(FLAGS_flash_attn)) {
        if (FLAGS_flash_attn == "on" || FLAGS_flash_attn == "1" || FLAGS_flash_attn == "true") {
            params.flash_attn_type = LHM_FLASH_ATTN_TYPE_ENABLED;
        } else if (FLAGS_flash_attn == "off" || FLAGS_flash_attn == "0" || FLAGS_flash_attn == "false") {
            params.flash_attn_type = LHM_FLASH_ATTN_TYPE_DISABLED;
        } else if (FLAGS_flash_attn == "auto") {
            params.flash_attn_type = LHM_FLASH_ATTN_TYPE_AUTO;
        } else {
            throw std::runtime_error(string_format("error: unknown value for --flash-attn: '%s'\n", FLAGS_flash_attn.c_str()));
        }
    }

    // ========================================================================
    // Prompt parameters
    // ========================================================================
    if (is_set(FLAGS_prompt)) {
        params.prompt = FLAGS_prompt;
    }

    if (is_set(FLAGS_system_prompt)) {
        params.system_prompt = FLAGS_system_prompt;
    }

    if (!FLAGS_perf) {
        params.no_perf = true;
        params.sampling.no_perf = true;
    }

    if (!FLAGS_show_timings) {
        params.show_timings = false;
    }

    // --file
    if (is_set(FLAGS_file)) {
        params.prompt = read_file_contents(FLAGS_file);
        params.prompt_file = FLAGS_file;
        if (!params.prompt.empty() && params.prompt.back() == '\n') {
            params.prompt.pop_back();
        }
    }

    // --system-prompt-file
    if (is_set(FLAGS_system_prompt_file)) {
        params.system_prompt = read_file_contents(FLAGS_system_prompt_file);
        if (!params.system_prompt.empty() && params.system_prompt.back() == '\n') {
            params.system_prompt.pop_back();
        }
    }

    // --in-file
    if (is_set(FLAGS_in_file)) {
        for (const auto & item : parse_csv_row(FLAGS_in_file)) {
            std::ifstream file(item);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", item.c_str()));
            }
            params.in_files.push_back(item);
        }
    }

    // --binary-file
    if (is_set(FLAGS_binary_file)) {
        std::ifstream file(FLAGS_binary_file, std::ios::binary);
        if (!file) {
            throw std::runtime_error(string_format("error: failed to open file '%s'\n", FLAGS_binary_file.c_str()));
        }
        params.prompt_file = FLAGS_binary_file;
        std::ostringstream ss;
        ss << file.rdbuf();
        params.prompt = ss.str();
    }

    if (!FLAGS_escape) {
        params.escape = false;
    }

    if (FLAGS_print_token_count != -1) {
        params.n_print = FLAGS_print_token_count;
    }

    // --prompt-cache
    if (is_set(FLAGS_prompt_cache)) {
        params.path_prompt_cache = FLAGS_prompt_cache;
    }

    if (FLAGS_prompt_cache_all) {
        params.prompt_cache_all = true;
    }

    if (FLAGS_prompt_cache_ro) {
        params.prompt_cache_ro = true;
    }

    // --reverse-prompt
    if (is_set(FLAGS_reverse_prompt)) {
        params.antiprompt.emplace_back(FLAGS_reverse_prompt);
    }

    if (FLAGS_special) {
        params.special = true;
    }

    if (FLAGS_conversation) {
        params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
    }

    if (FLAGS_single_turn) {
        params.single_turn = true;
    }

    if (FLAGS_interactive) {
        params.interactive = true;
    }

    if (FLAGS_interactive_first) {
        params.interactive_first = true;
    }

    if (FLAGS_multiline_input) {
        params.multiline_input = true;
    }

    if (FLAGS_in_prefix_bos) {
        params.input_prefix_bos = true;
        params.enable_chat_template = false;
    }

    if (is_set(FLAGS_in_prefix)) {
        params.input_prefix = FLAGS_in_prefix;
        params.enable_chat_template = false;
    }

    if (is_set(FLAGS_in_suffix)) {
        params.input_suffix = FLAGS_in_suffix;
        params.enable_chat_template = false;
    }

    if (!FLAGS_warmup) {
        params.warmup = false;
    }

    if (FLAGS_spm_infill) {
        params.spm_infill = true;
    }

    // ========================================================================
    // Sampling parameters
    // ========================================================================
    if (is_set(FLAGS_samplers)) {
        const auto sampler_names = string_split<std::string>(FLAGS_samplers, ';');
        params.sampling.samplers = common_sampler_types_from_names(sampler_names);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS;
    }

    if (is_set(FLAGS_seed)) {
        params.sampling.seed = std::stoul(FLAGS_seed);
    }

    if (is_set(FLAGS_sampler_seq)) {
        params.sampling.samplers = common_sampler_types_from_chars(FLAGS_sampler_seq);
    }

    if (FLAGS_ignore_eos) {
        params.sampling.ignore_eos = true;
    }

    if (is_set(FLAGS_temp)) {
        params.sampling.temp = std::stof(FLAGS_temp);
        params.sampling.temp = std::max(params.sampling.temp, 0.0f);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    }

    if (FLAGS_top_k != 40) {
        params.sampling.top_k = FLAGS_top_k;
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
    }

    if (is_set(FLAGS_top_p)) {
        params.sampling.top_p = std::stof(FLAGS_top_p);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
    }

    if (is_set(FLAGS_min_p)) {
        params.sampling.min_p = std::stof(FLAGS_min_p);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
    }

    if (is_set(FLAGS_top_nsigma)) {
        params.sampling.top_n_sigma = std::stof(FLAGS_top_nsigma);
    }

    if (is_set(FLAGS_xtc_probability)) {
        params.sampling.xtc_probability = std::stof(FLAGS_xtc_probability);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY;
    }

    if (is_set(FLAGS_xtc_threshold)) {
        params.sampling.xtc_threshold = std::stof(FLAGS_xtc_threshold);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
    }

    if (is_set(FLAGS_typical)) {
        params.sampling.typ_p = std::stof(FLAGS_typical);
    }

    if (FLAGS_repeat_last_n != 64) {
        if (FLAGS_repeat_last_n < -1) {
            throw std::runtime_error(string_format("error: invalid repeat-last-n = %d\n", FLAGS_repeat_last_n));
        }
        params.sampling.penalty_last_n = FLAGS_repeat_last_n;
        params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N;
    }

    if (is_set(FLAGS_repeat_penalty)) {
        params.sampling.penalty_repeat = std::stof(FLAGS_repeat_penalty);
        params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
    }

    if (is_set(FLAGS_presence_penalty)) {
        params.sampling.penalty_present = std::stof(FLAGS_presence_penalty);
    }

    if (is_set(FLAGS_frequency_penalty)) {
        params.sampling.penalty_freq = std::stof(FLAGS_frequency_penalty);
    }

    if (is_set(FLAGS_dry_multiplier)) {
        params.sampling.dry_multiplier = std::stof(FLAGS_dry_multiplier);
    }

    if (is_set(FLAGS_dry_base)) {
        float potential_base = std::stof(FLAGS_dry_base);
        if (potential_base >= 1.0f) {
            params.sampling.dry_base = potential_base;
        }
    }

    if (FLAGS_dry_allowed_length != 2) {
        params.sampling.dry_allowed_length = FLAGS_dry_allowed_length;
    }

    if (FLAGS_dry_penalty_last_n != -1) {
        if (FLAGS_dry_penalty_last_n < -1) {
            throw std::runtime_error(string_format("error: invalid dry-penalty-last-n = %d\n", FLAGS_dry_penalty_last_n));
        }
        params.sampling.dry_penalty_last_n = FLAGS_dry_penalty_last_n;
    }

    if (is_set(FLAGS_dry_sequence_breaker)) {
        // Parse dry sequence breakers
        if (FLAGS_dry_sequence_breaker == "none") {
            params.sampling.dry_sequence_breakers.clear();
        } else {
            params.sampling.dry_sequence_breakers.clear();
            for (const auto & sb : string_split<std::string>(FLAGS_dry_sequence_breaker, ',')) {
                params.sampling.dry_sequence_breakers.push_back(sb);
            }
        }
    }

    if (is_set(FLAGS_adaptive_target)) {
        params.sampling.adaptive_target = std::stof(FLAGS_adaptive_target);
    }

    if (is_set(FLAGS_adaptive_decay)) {
        params.sampling.adaptive_decay = std::stof(FLAGS_adaptive_decay);
    }

    if (is_set(FLAGS_dynatemp_range)) {
        params.sampling.dynatemp_range = std::stof(FLAGS_dynatemp_range);
    }

    if (is_set(FLAGS_dynatemp_exp)) {
        params.sampling.dynatemp_exponent = std::stof(FLAGS_dynatemp_exp);
    }

    if (FLAGS_mirostat != 0) {
        params.sampling.mirostat = FLAGS_mirostat;
    }

    if (is_set(FLAGS_mirostat_lr)) {
        params.sampling.mirostat_eta = std::stof(FLAGS_mirostat_lr);
    }

    if (is_set(FLAGS_mirostat_ent)) {
        params.sampling.mirostat_tau = std::stof(FLAGS_mirostat_ent);
    }

    // --logit-bias
    if (is_set(FLAGS_logit_bias)) {
        // Parse TOKEN_ID(+/-)BIAS format
        // For simplicity, store as string to be parsed later
        // The original arg.h handler does: params.sampling.logit_bias.push_back(...)
        // We need to parse the format "TOKEN_ID(+/-)BIAS"
        std::string bias_str = FLAGS_logit_bias;
        // Parse format: e.g. "123+1.0" or "456-0.5"
        size_t pos = bias_str.find_first_of("+-");
        if (pos != std::string::npos && pos > 0) {
            lhm_token tok = std::stoi(bias_str.substr(0, pos));
            float bias = std::stof(bias_str.substr(pos));
            params.sampling.logit_bias.push_back({tok, bias});
        }
    }

    // --grammar
    if (is_set(FLAGS_grammar)) {
        params.sampling.grammar = FLAGS_grammar;
    }

    // --grammar-file
    if (is_set(FLAGS_grammar_file)) {
        params.sampling.grammar = read_file_contents(FLAGS_grammar_file);
    }

    // --json-schema
    if (is_set(FLAGS_json_schema)) {
        params.sampling.grammar = FLAGS_json_schema; // will be converted later
    }

    // --json-schema-file
    if (is_set(FLAGS_json_schema_file)) {
        params.sampling.grammar = read_file_contents(FLAGS_json_schema_file);
    }

    if (FLAGS_backend_sampling) {
        params.sampling.backend_sampling = true;
    }

    // ========================================================================
    // Model type parameters
    // ========================================================================
    if (is_set(FLAGS_pooling)) {
        if (FLAGS_pooling == "mean") {
            params.pooling_type = LHM_POOLING_TYPE_MEAN;
        } else if (FLAGS_pooling == "cls") {
            params.pooling_type = LHM_POOLING_TYPE_CLS;
        } else if (FLAGS_pooling == "last") {
            params.pooling_type = LHM_POOLING_TYPE_LAST;
        } else {
            throw std::invalid_argument(string_format("unknown pooling type: %s", FLAGS_pooling.c_str()));
        }
    }

    if (is_set(FLAGS_attention)) {
        if (FLAGS_attention == "causal") {
            params.attention_type = LHM_ATTENTION_TYPE_CAUSAL;
        } else if (FLAGS_attention == "non-causal") {
            params.attention_type = LHM_ATTENTION_TYPE_NON_CAUSAL;
        } else {
            throw std::invalid_argument(string_format("unknown attention type: %s", FLAGS_attention.c_str()));
        }
    }

    if (is_set(FLAGS_rope_scaling)) {
        if (FLAGS_rope_scaling == "none") {
            params.rope_scaling_type = LHM_ROPE_SCALING_TYPE_NONE;
        } else if (FLAGS_rope_scaling == "linear") {
            params.rope_scaling_type = LHM_ROPE_SCALING_TYPE_LINEAR;
        } else if (FLAGS_rope_scaling == "yarn") {
            params.rope_scaling_type = LHM_ROPE_SCALING_TYPE_YARN;
        } else {
            throw std::invalid_argument(string_format("unknown rope scaling type: %s", FLAGS_rope_scaling.c_str()));
        }
    }

    if (is_set(FLAGS_rope_scale)) {
        params.rope_freq_scale = std::stof(FLAGS_rope_scale);
    }

    if (is_set(FLAGS_rope_freq_base)) {
        params.rope_freq_base = std::stof(FLAGS_rope_freq_base);
    }

    if (is_set(FLAGS_rope_freq_scale)) {
        params.rope_freq_scale = std::stof(FLAGS_rope_freq_scale);
    }

    if (FLAGS_yarn_orig_ctx != 0) {
        params.yarn_orig_ctx = FLAGS_yarn_orig_ctx;
    }

    if (is_set(FLAGS_yarn_ext_factor)) {
        params.yarn_ext_factor = std::stof(FLAGS_yarn_ext_factor);
    }

    if (is_set(FLAGS_yarn_attn_factor)) {
        params.yarn_attn_factor = std::stof(FLAGS_yarn_attn_factor);
    }

    if (is_set(FLAGS_yarn_beta_slow)) {
        params.yarn_beta_slow = std::stof(FLAGS_yarn_beta_slow);
    }

    if (is_set(FLAGS_yarn_beta_fast)) {
        params.yarn_beta_fast = std::stof(FLAGS_yarn_beta_fast);
    }

    if (FLAGS_grp_attn_n != 1) {
        params.grp_attn_n = FLAGS_grp_attn_n;
    }

    if (FLAGS_grp_attn_w != 512) {
        params.grp_attn_w = FLAGS_grp_attn_w;
    }

    // ========================================================================
    // KV cache / offload parameters
    // ========================================================================
    if (!FLAGS_kv_offload) {
        params.no_kv_offload = true;
    }

    if (!FLAGS_repack) {
        params.no_extra_bufts = true;
    }

    if (FLAGS_no_host) {
        params.no_host = true;
    }

    if (is_set(FLAGS_cache_type_k) && FLAGS_cache_type_k != "f16") {
        // Parse ggml_type from string
        params.cache_type_k = ggml_type_from_name(FLAGS_cache_type_k);
    }

    if (is_set(FLAGS_cache_type_v) && FLAGS_cache_type_v != "f16") {
        params.cache_type_v = ggml_type_from_name(FLAGS_cache_type_v);
    }

    if (is_set(FLAGS_defrag_thold)) {
        // Store defrag threshold - handled during model loading
    }

    // ========================================================================
    // Parallel / batching
    // ========================================================================
    if (FLAGS_parallel != 1) {
        params.n_parallel = FLAGS_parallel;
    }

    if (FLAGS_sequences != 1) {
        params.n_sequences = FLAGS_sequences;
    }

    if (!FLAGS_cont_batching) {
        params.cont_batching = false;
    }

    // ========================================================================
    // Multimodal parameters
    // ========================================================================
    if (is_set(FLAGS_mmproj)) {
        params.mmproj.path = FLAGS_mmproj;
    }

    if (is_set(FLAGS_mmproj_url)) {
        params.mmproj.url = FLAGS_mmproj_url;
    }

    if (!FLAGS_mmproj_auto) {
        params.no_mmproj = true;
    }

    if (!FLAGS_mmproj_offload) {
        params.mmproj_use_gpu = false;
    }

    if (is_set(FLAGS_image)) {
        for (const auto & item : parse_csv_row(FLAGS_image)) {
            params.image.push_back(item);
        }
    }

    if (FLAGS_image_min_tokens != -1) {
        params.image_min_tokens = FLAGS_image_min_tokens;
    }

    if (FLAGS_image_max_tokens != -1) {
        params.image_max_tokens = FLAGS_image_max_tokens;
    }

    // ========================================================================
    // RPC / memory parameters
    // ========================================================================
    if (is_set(FLAGS_rpc)) {
        // RPC servers handled during model loading
    }

    if (FLAGS_mlock) {
        params.use_mlock = true;
    }

    if (!FLAGS_mmap) {
        params.use_mmap = false;
    }

    if (FLAGS_direct_io) {
        params.use_direct_io = true;
    }

    if (is_set(FLAGS_numa)) {
        if (FLAGS_numa == "distribute") {
            params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
        } else if (FLAGS_numa == "isolate") {
            params.numa = GGML_NUMA_STRATEGY_ISOLATE;
        } else if (FLAGS_numa == "numactl") {
            params.numa = GGML_NUMA_STRATEGY_NUMACTL;
        } else if (FLAGS_numa == "1") {
            params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
        } else {
            throw std::invalid_argument(string_format("unknown NUMA strategy: %s", FLAGS_numa.c_str()));
        }
    }

    if (is_set(FLAGS_device)) {
        // Device selection handled during model loading
    }

    // --override-tensor
    if (is_set(FLAGS_override_tensor)) {
        parse_tensor_buffer_overrides(FLAGS_override_tensor, params.tensor_buft_overrides);
    }

    if (FLAGS_cpu_moe) {
        // Add CPU MoE override
        params.tensor_buft_overrides.push_back({nullptr, nullptr}); // placeholder
    }

    if (FLAGS_n_cpu_moe > 0) {
        // Add N CPU MoE overrides
        for (int i = 0; i < FLAGS_n_cpu_moe; i++) {
            params.tensor_buft_overrides.push_back({nullptr, nullptr}); // placeholder
        }
    }

    // ========================================================================
    // GPU parameters
    // ========================================================================
    if (is_set(FLAGS_gpu_layers)) {
        params.n_gpu_layers = std::stoi(FLAGS_gpu_layers);
    }

    if (is_set(FLAGS_split_mode)) {
        if (FLAGS_split_mode == "layer") {
            params.split_mode = LHM_SPLIT_MODE_LAYER;
        } else if (FLAGS_split_mode == "row") {
            params.split_mode = LHM_SPLIT_MODE_ROW;
        } else if (FLAGS_split_mode == "none") {
            params.split_mode = LHM_SPLIT_MODE_NONE;
        } else {
            throw std::invalid_argument(string_format("unknown split mode: %s", FLAGS_split_mode.c_str()));
        }
    }

    if (is_set(FLAGS_tensor_split)) {
        // Parse tensor split values
        std::istringstream iss(FLAGS_tensor_split);
        std::string token;
        int idx = 0;
        while (std::getline(iss, token, ',') && idx < 128) {
            params.tensor_split[idx++] = std::stof(token);
        }
    }

    if (FLAGS_main_gpu != 0) {
        params.main_gpu = FLAGS_main_gpu;
    }

    // ========================================================================
    // Fit parameters
    // ========================================================================
    if (is_set(FLAGS_fit)) {
        params.fit_params = parse_bool_string(FLAGS_fit);
    }

    if (is_set(FLAGS_fit_print)) {
        params.fit_params_print = parse_bool_string(FLAGS_fit_print);
    }

    if (is_set(FLAGS_fit_target)) {
        for (const auto & item : parse_csv_row(FLAGS_fit_target)) {
            params.fit_params_target.push_back(std::stoull(item));
        }
    }

    if (FLAGS_fit_ctx != 4096) {
        params.fit_params_min_ctx = FLAGS_fit_ctx;
    }

    if (FLAGS_check_tensors) {
        params.check_tensors = true;
    }

    // ========================================================================
    // Override / LoRA / control vector parameters
    // ========================================================================
    if (is_set(FLAGS_override_kv)) {
        // Parse key=value pairs
        // Format: KEY=TYPE:VALUE
        // For simplicity, store as string to be parsed later
    }

    if (FLAGS_op_offload) {
        params.no_op_offload = true;
    }

    // --lora
    if (is_set(FLAGS_lora)) {
        common_adapter_lora_info lora;
        lora.path = FLAGS_lora;
        lora.scale = 1.0f;
        params.lora_adapters.push_back(lora);
    }

    // --lora-scaled (format: PATH SCALE)
    if (is_set(FLAGS_lora_scaled)) {
        auto parts = string_split<std::string>(FLAGS_lora_scaled, ' ');
        if (parts.size() >= 2) {
            common_adapter_lora_info lora;
            lora.path = parts[0];
            lora.scale = std::stof(parts[1]);
            params.lora_adapters.push_back(lora);
        }
    }

    // --control-vector
    if (is_set(FLAGS_control_vector)) {
        common_control_vector_load_info cv;
        cv.path = FLAGS_control_vector;
        cv.scale = 1.0f;
        params.control_vectors.push_back(cv);
    }

    // --control-vector-scaled (format: PATH SCALE)
    if (is_set(FLAGS_control_vector_scaled)) {
        auto parts = string_split<std::string>(FLAGS_control_vector_scaled, ' ');
        if (parts.size() >= 2) {
            common_control_vector_load_info cv;
            cv.path = parts[0];
            cv.scale = std::stof(parts[1]);
            params.control_vectors.push_back(cv);
        }
    }

    // --control-vector-layer-range (format: START END)
    if (is_set(FLAGS_control_vector_layer_range)) {
        auto parts = string_split<std::string>(FLAGS_control_vector_layer_range, ' ');
        if (parts.size() >= 2) {
            params.control_vector_layer_start = std::stoi(parts[0]);
            params.control_vector_layer_end = std::stoi(parts[1]);
        }
    }

    if (FLAGS_lora_init_without_apply) {
        params.lora_init_without_apply = true;
    }

    // ========================================================================
    // Model path parameters
    // ========================================================================
    if (is_set(FLAGS_alias)) {
        params.model_alias.insert(FLAGS_alias);
    }

    if (is_set(FLAGS_tags)) {
        params.model_tags.insert(FLAGS_tags);
    }

    if (is_set(FLAGS_model)) {
        params.model.path = FLAGS_model;
    }

    if (is_set(FLAGS_model_url)) {
        params.model.url = FLAGS_model_url;
    }

    if (is_set(FLAGS_docker_repo)) {
        params.model.docker_repo = FLAGS_docker_repo;
    }

    if (is_set(FLAGS_hf_repo)) {
        params.model.hf_repo = FLAGS_hf_repo;
    }

    if (is_set(FLAGS_hf_file)) {
        params.model.hf_file = FLAGS_hf_file;
    }

    if (is_set(FLAGS_hf_repo_v)) {
        params.vocoder.model.hf_repo = FLAGS_hf_repo_v;
    }

    if (is_set(FLAGS_hf_file_v)) {
        params.vocoder.model.hf_file = FLAGS_hf_file_v;
    }

    if (is_set(FLAGS_hf_token)) {
        params.hf_token = FLAGS_hf_token;
    }

    // ========================================================================
    // Retrieval / embedding parameters
    // ========================================================================
    if (is_set(FLAGS_context_file)) {
        for (const auto & item : parse_csv_row(FLAGS_context_file)) {
            params.context_files.push_back(item);
        }
    }

    if (FLAGS_chunk_size != 64) {
        params.chunk_size = FLAGS_chunk_size;
    }

    if (is_set(FLAGS_chunk_separator) && FLAGS_chunk_separator != "\n") {
        params.chunk_separator = FLAGS_chunk_separator;
    }

    if (FLAGS_embd_normalize != 2) {
        params.embd_normalize = FLAGS_embd_normalize;
    }

    if (is_set(FLAGS_embd_output_format)) {
        params.embd_out = FLAGS_embd_output_format;
    }

    if (is_set(FLAGS_embd_separator) && FLAGS_embd_separator != "\n") {
        params.embd_sep = FLAGS_embd_separator;
    }

    if (is_set(FLAGS_cls_separator) && FLAGS_cls_separator != "\t") {
        params.cls_sep = FLAGS_cls_separator;
    }

    if (FLAGS_embedding) {
        params.embedding = true;
    }

    if (FLAGS_rerank) {
        params.embedding = true;
        params.pooling_type = LHM_POOLING_TYPE_RANK;
    }

    // ========================================================================
    // Passkey / benchmark parameters
    // ========================================================================
    if (FLAGS_junk != 250) {
        params.n_junk = FLAGS_junk;
    }

    if (FLAGS_pos != -1) {
        params.i_pos = FLAGS_pos;
    }

    // ========================================================================
    // Imatrix parameters
    // ========================================================================
    if (is_set(FLAGS_output)) {
        params.out_file = FLAGS_output;
    }

    if (FLAGS_output_frequency != 10) {
        params.n_out_freq = FLAGS_output_frequency;
    }

    if (is_set(FLAGS_output_format)) {
        // Could be imatrix format or batched_bench format
        if (FLAGS_output_format == "dat") {
            params.imat_dat = 1;
        }
    }

    if (FLAGS_save_frequency != 0) {
        params.n_save_freq = FLAGS_save_frequency;
    }

    if (FLAGS_process_output) {
        params.process_output = true;
    }

    if (!FLAGS_ppl) {
        params.compute_ppl = false;
    }

    if (FLAGS_chunk != 0) {
        params.i_chunk = FLAGS_chunk;
    }

    if (FLAGS_show_statistics) {
        params.show_statistics = true;
    }

    if (FLAGS_parse_special) {
        params.parse_special = true;
    }

    // ========================================================================
    // Bench parameters
    // ========================================================================
    if (FLAGS_pps) {
        params.is_pp_shared = true;
    }

    if (FLAGS_tgs) {
        params.is_tg_separate = true;
    }

    if (is_set(FLAGS_npp)) {
        for (const auto & v : string_split<std::string>(FLAGS_npp, ',')) {
            params.n_pp.push_back(std::stoi(v));
        }
    }

    if (is_set(FLAGS_ntg)) {
        for (const auto & v : string_split<std::string>(FLAGS_ntg, ',')) {
            params.n_tg.push_back(std::stoi(v));
        }
    }

    if (is_set(FLAGS_npl)) {
        for (const auto & v : string_split<std::string>(FLAGS_npl, ',')) {
            params.n_pl.push_back(std::stoi(v));
        }
    }

    // ========================================================================
    // Perplexity parameters
    // ========================================================================
    if (FLAGS_ppl_stride != 0) {
        params.ppl_stride = FLAGS_ppl_stride;
    }

    if (FLAGS_ppl_output_type != 0) {
        params.ppl_output_type = FLAGS_ppl_output_type;
    }

    if (FLAGS_hellaswag) {
        params.hellaswag = true;
    }

    if (FLAGS_hellaswag_tasks != 400) {
        params.hellaswag_tasks = FLAGS_hellaswag_tasks;
    }

    if (FLAGS_winogrande) {
        params.winogrande = true;
    }

    if (FLAGS_winogrande_tasks != 0) {
        params.winogrande_tasks = FLAGS_winogrande_tasks;
    }

    if (FLAGS_multiple_choice) {
        params.multiple_choice = true;
    }

    if (FLAGS_multiple_choice_tasks != 0) {
        params.multiple_choice_tasks = FLAGS_multiple_choice_tasks;
    }

    if (FLAGS_kl_divergence) {
        params.kl_divergence = true;
    }

    if (is_set(FLAGS_save_all_logits)) {
        params.logits_file = FLAGS_save_all_logits;
    }

    // ========================================================================
    // Server parameters
    // ========================================================================
    if (is_set(FLAGS_host) && FLAGS_host != "127.0.0.1") {
        params.hostname = FLAGS_host;
    }

    if (FLAGS_port != 8080) {
        params.port = FLAGS_port;
    }

    if (FLAGS_reuse_port) {
        params.reuse_port = true;
    }

    if (is_set(FLAGS_path)) {
        params.public_path = FLAGS_path;
    }

    if (is_set(FLAGS_api_prefix)) {
        params.api_prefix = FLAGS_api_prefix;
    }

    if (is_set(FLAGS_webui_config)) {
        params.webui_config_json = FLAGS_webui_config;
        params.ui_config_json = FLAGS_webui_config;
    }

    if (is_set(FLAGS_ui_config)) {
        params.ui_config_json = FLAGS_ui_config;
        params.webui_config_json = FLAGS_ui_config;
    }

    if (is_set(FLAGS_webui_config_file)) {
        params.webui_config_json = read_file_contents(FLAGS_webui_config_file);
        params.ui_config_json = params.webui_config_json;
    }

    if (is_set(FLAGS_ui_config_file)) {
        params.ui_config_json = read_file_contents(FLAGS_ui_config_file);
        params.webui_config_json = params.ui_config_json;
    }

    if (FLAGS_webui_mcp_proxy) {
        params.webui_mcp_proxy = true;
        params.ui_mcp_proxy = true;
    }

    if (FLAGS_ui_mcp_proxy) {
        params.ui_mcp_proxy = true;
        params.webui_mcp_proxy = true;
    }

    if (is_set(FLAGS_tools)) {
        for (const auto & t : string_split<std::string>(FLAGS_tools, ',')) {
            params.server_tools.push_back(t);
        }
    }

    if (!FLAGS_webui) {
        params.webui = false;
        params.ui = false;
    }

    if (!FLAGS_ui) {
        params.ui = false;
        params.webui = false;
    }

    if (FLAGS_embedding) {
        params.embedding = true;
    }

    // --api-key
    if (is_set(FLAGS_api_key)) {
        params.api_keys.push_back(FLAGS_api_key);
    }

    // --api-key-file
    if (is_set(FLAGS_api_key_file)) {
        std::string key = read_file_contents(FLAGS_api_key_file);
        // Trim whitespace
        while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' ')) {
            key.pop_back();
        }
        params.api_keys.push_back(key);
    }

    if (is_set(FLAGS_ssl_key_file)) {
        params.ssl_file_key = FLAGS_ssl_key_file;
    }

    if (is_set(FLAGS_ssl_cert_file)) {
        params.ssl_file_cert = FLAGS_ssl_cert_file;
    }

    if (is_set(FLAGS_chat_template_kwargs)) {
        // Parse JSON kwargs
        // For simplicity, store as string
    }

    if (FLAGS_timeout != 3600) {
        params.timeout_read = FLAGS_timeout;
        params.timeout_write = FLAGS_timeout;
    }

    if (FLAGS_sse_ping_interval != 30) {
        params.sse_ping_interval = FLAGS_sse_ping_interval;
    }

    if (FLAGS_threads_http != -1) {
        params.n_threads_http = FLAGS_threads_http;
    }

    if (!FLAGS_cache_prompt) {
        params.cache_prompt = false;
    }

    if (FLAGS_cache_reuse != 0) {
        params.n_cache_reuse = FLAGS_cache_reuse;
    }

    if (FLAGS_metrics) {
        params.endpoint_metrics = true;
    }

    if (FLAGS_props) {
        params.endpoint_props = true;
    }

    if (!FLAGS_slots) {
        params.endpoint_slots = false;
    }

    if (is_set(FLAGS_slot_save_path)) {
        params.slot_save_path = FLAGS_slot_save_path;
    }

    if (is_set(FLAGS_media_path)) {
        params.media_path = FLAGS_media_path;
    }

    if (is_set(FLAGS_models_dir)) {
        params.models_dir = FLAGS_models_dir;
    }

    if (is_set(FLAGS_models_preset)) {
        params.models_preset = FLAGS_models_preset;
    }

    if (FLAGS_models_max != 4) {
        params.models_max = FLAGS_models_max;
    }

    if (!FLAGS_models_autoload) {
        params.models_autoload = false;
    }

    if (!FLAGS_jinja) {
        params.use_jinja = false;
    }

    if (is_set(FLAGS_reasoning_format)) {
        if (FLAGS_reasoning_format == "deepseek") {
            params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        } else if (FLAGS_reasoning_format == "none") {
            params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        }
    }

    // --reasoning (on/off/auto)
    if (is_set(FLAGS_reasoning)) {
        if (FLAGS_reasoning == "on" || FLAGS_reasoning == "1" || FLAGS_reasoning == "true") {
            params.enable_reasoning = 1;
        } else if (FLAGS_reasoning == "off" || FLAGS_reasoning == "0" || FLAGS_reasoning == "false") {
            params.enable_reasoning = 0;
        } else if (FLAGS_reasoning == "auto") {
            params.enable_reasoning = -1;
        }
    }

    if (FLAGS_reasoning_budget >= 0) {
        params.sampling.reasoning_budget_tokens = FLAGS_reasoning_budget;
    }

    if (is_set(FLAGS_reasoning_budget_message)) {
        params.sampling.reasoning_budget_message = FLAGS_reasoning_budget_message;
    }

    if (is_set(FLAGS_chat_template)) {
        params.chat_template = FLAGS_chat_template;
    }

    if (is_set(FLAGS_chat_template_file)) {
        params.chat_template = read_file_contents(FLAGS_chat_template_file);
    }

    if (FLAGS_skip_chat_parsing) {
        params.force_pure_content_parser = true;
    }

    if (!FLAGS_prefill_assistant) {
        params.prefill_assistant = false;
    }

    if (is_set(FLAGS_slot_prompt_similarity)) {
        params.slot_prompt_similarity = std::stof(FLAGS_slot_prompt_similarity);
    }

    if (FLAGS_sleep_idle_seconds >= 0) {
        params.sleep_idle_seconds = FLAGS_sleep_idle_seconds;
    }

    if (FLAGS_simple_io) {
        params.simple_io = true;
    }

    // ========================================================================
    // Control vector generator parameters
    // ========================================================================
    if (is_set(FLAGS_positive_file)) {
        params.cvector_positive_file = FLAGS_positive_file;
    }

    if (is_set(FLAGS_negative_file)) {
        params.cvector_negative_file = FLAGS_negative_file;
    }

    if (FLAGS_pca_batch != 100) {
        params.n_pca_batch = FLAGS_pca_batch;
    }

    if (FLAGS_pca_iter != 1000) {
        params.n_pca_iterations = FLAGS_pca_iter;
    }

    if (is_set(FLAGS_method)) {
        if (FLAGS_method == "pca") {
            params.cvector_dimre_method = DIMRE_METHOD_PCA;
        }
    }

    // ========================================================================
    // Logging parameters (server-specific)
    // ========================================================================
    if (FLAGS_verbose) {
        params.verbosity = 4; // LOG_LEVEL_DEBUG
    }

    if (FLAGS_offline) {
        params.offline = true;
    }

    if (FLAGS_verbosity != 3) {
        params.verbosity = FLAGS_verbosity;
    }

    if (FLAGS_log_disable) {
        params.log_disable = true;
    }

    if (FLAGS_log_file_flag) {
        params.log_file = true;
    }

    if (is_set(FLAGS_log_file)) {
        params.log_file_path = FLAGS_log_file;
    }

    if (!FLAGS_log_prefix) {
        params.log_prefix = false;
    }

    if (!FLAGS_log_timestamps) {
        params.log_timestamps = false;
    }

    if (!FLAGS_log_colors) {
        params.use_color = false;
    }

    if (is_set(FLAGS_log_level)) {
        // log_level is a string like "debug", "info", "warn", "error"
        params.verbosity = 3; // default
        const auto & level = FLAGS_log_level;
        if (level == "debug") params.verbosity = 4;
        else if (level == "info") params.verbosity = 3;
        else if (level == "warn" || level == "warning") params.verbosity = 2;
        else if (level == "error") params.verbosity = 1;
        else if (level == "disable" || level == "none") params.verbosity = 0;
    }

    if (is_set(FLAGS_log_pattern)) {
        // log_pattern is a string, store for later use
        params.log_pattern = FLAGS_log_pattern;
    }

    if (is_set(FLAGS_log_prompts_dir)) {
        params.path_prompts_log_dir = FLAGS_log_prompts_dir;
    }

    // ========================================================================
    // Speculative decoding parameters
    // ========================================================================
    if (is_set(FLAGS_spec_draft_hf)) {
        params.speculative.draft.mparams.hf_repo = FLAGS_spec_draft_hf;
    }

    if (FLAGS_spec_draft_threads != 0) {
        params.speculative.draft.cpuparams.n_threads = FLAGS_spec_draft_threads;
        if (params.speculative.draft.cpuparams.n_threads <= 0) {
            params.speculative.draft.cpuparams.n_threads = std::thread::hardware_concurrency();
        }
    }

    if (FLAGS_spec_draft_threads_batch != 0) {
        params.speculative.draft.cpuparams_batch.n_threads = FLAGS_spec_draft_threads_batch;
        if (params.speculative.draft.cpuparams_batch.n_threads <= 0) {
            params.speculative.draft.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
        }
    }

    if (is_set(FLAGS_spec_draft_cpu_mask)) {
        params.speculative.draft.cpuparams.mask_valid = true;
        if (!parse_cpu_mask(FLAGS_spec_draft_cpu_mask, params.speculative.draft.cpuparams.cpumask)) {
            throw std::invalid_argument("invalid cpumask for draft model");
        }
    }

    if (is_set(FLAGS_spec_draft_cpu_range)) {
        params.speculative.draft.cpuparams.mask_valid = true;
        if (!parse_cpu_range(FLAGS_spec_draft_cpu_range, params.speculative.draft.cpuparams.cpumask)) {
            throw std::invalid_argument("invalid range for draft model");
        }
    }

    if (FLAGS_spec_draft_cpu_strict >= 0) {
        params.speculative.draft.cpuparams.strict_cpu = FLAGS_spec_draft_cpu_strict;
    }

    if (FLAGS_spec_draft_prio >= 0) {
        params.speculative.draft.cpuparams.priority = (enum ggml_sched_priority) FLAGS_spec_draft_prio;
    }

    if (FLAGS_spec_draft_poll >= 0) {
        params.speculative.draft.cpuparams.poll = FLAGS_spec_draft_poll;
    }

    if (is_set(FLAGS_spec_draft_cpu_mask_batch)) {
        params.speculative.draft.cpuparams_batch.mask_valid = true;
        if (!parse_cpu_mask(FLAGS_spec_draft_cpu_mask_batch, params.speculative.draft.cpuparams_batch.cpumask)) {
            throw std::invalid_argument("invalid cpumask for draft model batch");
        }
    }

    if (FLAGS_spec_draft_cpu_range_batch) {
        // Parse CPU range for draft model batch
    }

    if (FLAGS_spec_draft_cpu_strict_batch >= 0) {
        params.speculative.draft.cpuparams_batch.strict_cpu = FLAGS_spec_draft_cpu_strict_batch;
    }

    if (FLAGS_spec_draft_prio_batch >= 0) {
        params.speculative.draft.cpuparams_batch.priority = (enum ggml_sched_priority) FLAGS_spec_draft_prio_batch;
    }

    if (FLAGS_spec_draft_poll_batch >= 0) {
        params.speculative.draft.cpuparams_batch.poll = FLAGS_spec_draft_poll_batch;
    }

    if (is_set(FLAGS_spec_draft_type_k)) {
        params.speculative.draft.cache_type_k = ggml_type_from_name(FLAGS_spec_draft_type_k);
    }

    if (is_set(FLAGS_spec_draft_type_v)) {
        params.speculative.draft.cache_type_v = ggml_type_from_name(FLAGS_spec_draft_type_v);
    }

    if (is_set(FLAGS_spec_draft_override_tensor)) {
        parse_tensor_buffer_overrides(FLAGS_spec_draft_override_tensor, params.speculative.draft.tensor_buft_overrides);
    }

    if (FLAGS_spec_draft_cpu_moe) {
        params.speculative.draft.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (FLAGS_spec_draft_n_cpu_moe > 0) {
        for (int i = 0; i < FLAGS_spec_draft_n_cpu_moe; i++) {
            params.speculative.draft.tensor_buft_overrides.push_back({nullptr, nullptr});
        }
    }

    if (FLAGS_spec_draft_n_max != 0) {
        params.speculative.draft.n_max = FLAGS_spec_draft_n_max;
    }

    if (FLAGS_spec_draft_n_min != 0) {
        params.speculative.draft.n_min = FLAGS_spec_draft_n_min;
    }

    if (is_set(FLAGS_spec_draft_p_split)) {
        params.speculative.draft.p_split = std::stof(FLAGS_spec_draft_p_split);
    }

    if (is_set(FLAGS_spec_draft_p_min)) {
        params.speculative.draft.p_min = std::stof(FLAGS_spec_draft_p_min);
    }

    if (FLAGS_spec_draft_backend_sampling) {
        params.speculative.draft.backend_sampling = true;
    }

    if (is_set(FLAGS_spec_draft_device)) {
        // Device selection for draft model
    }

    if (is_set(FLAGS_spec_draft_ngl)) {
        params.speculative.draft.n_gpu_layers = std::stoi(FLAGS_spec_draft_ngl);
    }

    if (is_set(FLAGS_spec_draft_model)) {
        params.speculative.draft.mparams.path = FLAGS_spec_draft_model;
    }

    if (is_set(FLAGS_spec_type)) {
        for (const auto & t : string_split<std::string>(FLAGS_spec_type, ',')) {
            params.speculative.types.insert(t);
        }
    }

    if (FLAGS_spec_ngram_mod_n_min != 0) {
        params.speculative.ngram_mod.n_min = FLAGS_spec_ngram_mod_n_min;
    }

    if (FLAGS_spec_ngram_mod_n_max != 0) {
        params.speculative.ngram_mod.n_max = FLAGS_spec_ngram_mod_n_max;
    }

    if (FLAGS_spec_ngram_mod_n_match != 0) {
        params.speculative.ngram_mod.n_match = FLAGS_spec_ngram_mod_n_match;
    }

    if (FLAGS_spec_ngram_simple_size_n != 0) {
        params.speculative.ngram_simple.size_n = FLAGS_spec_ngram_simple_size_n;
    }

    if (FLAGS_spec_ngram_simple_size_m != 0) {
        params.speculative.ngram_simple.size_m = FLAGS_spec_ngram_simple_size_m;
    }

    if (FLAGS_spec_ngram_simple_min_hits != 0) {
        params.speculative.ngram_simple.min_hits = FLAGS_spec_ngram_simple_min_hits;
    }

    if (FLAGS_spec_ngram_map_k_size_n != 0) {
        params.speculative.ngram_map_k.size_n = FLAGS_spec_ngram_map_k_size_n;
    }

    if (FLAGS_spec_ngram_map_k_size_m != 0) {
        params.speculative.ngram_map_k.size_m = FLAGS_spec_ngram_map_k_size_m;
    }

    if (FLAGS_spec_ngram_map_k_min_hits != 0) {
        params.speculative.ngram_map_k.min_hits = FLAGS_spec_ngram_map_k_min_hits;
    }

    if (FLAGS_spec_ngram_map_k4v_size_n != 0) {
        params.speculative.ngram_map_k4v.size_n = FLAGS_spec_ngram_map_k4v_size_n;
    }

    if (FLAGS_spec_ngram_map_k4v_size_m != 0) {
        params.speculative.ngram_map_k4v.size_m = FLAGS_spec_ngram_map_k4v_size_m;
    }

    if (FLAGS_spec_ngram_map_k4v_min_hits != 0) {
        params.speculative.ngram_map_k4v.min_hits = FLAGS_spec_ngram_map_k4v_min_hits;
    }

    // ========================================================================
    // Lookup cache parameters
    // ========================================================================
    if (is_set(FLAGS_lookup_cache_static)) {
        params.speculative.ngram_cache.lookup_cache_static = FLAGS_lookup_cache_static;
    }

    if (is_set(FLAGS_lookup_cache_dynamic)) {
        params.speculative.ngram_cache.lookup_cache_dynamic = FLAGS_lookup_cache_dynamic;
    }

    // ========================================================================
    // Vocoder parameters
    // ========================================================================
    if (is_set(FLAGS_model_vocoder)) {
        params.vocoder.model.path = FLAGS_model_vocoder;
    }

    if (FLAGS_tts_use_guide_tokens) {
        params.vocoder.use_guide_tokens = true;
    }

    if (is_set(FLAGS_tts_speaker_file)) {
        params.vocoder.speaker_file = FLAGS_tts_speaker_file;
    }

    // ========================================================================
    // Diffusion parameters
    // ========================================================================
    if (FLAGS_diffusion_steps != 0) {
        params.diffusion.steps = FLAGS_diffusion_steps;
    }

    if (FLAGS_diffusion_visual) {
        params.diffusion.visual_mode = true;
    }

    if (is_set(FLAGS_diffusion_eps)) {
        params.diffusion.eps = FLAGS_diffusion_eps;
    }

    if (FLAGS_diffusion_algorithm != 0) {
        params.diffusion.algorithm = FLAGS_diffusion_algorithm;
    }

    if (is_set(FLAGS_diffusion_alg_temp)) {
        params.diffusion.alg_temp = std::stof(FLAGS_diffusion_alg_temp);
    }

    if (FLAGS_diffusion_block_length != 0) {
        params.diffusion.block_length = FLAGS_diffusion_block_length;
    }

    if (is_set(FLAGS_diffusion_cfg_scale)) {
        params.diffusion.cfg_scale = std::stof(FLAGS_diffusion_cfg_scale);
    }

    if (is_set(FLAGS_diffusion_add_gumbel_noise)) {
        params.diffusion.add_gumbel_noise = std::stof(FLAGS_diffusion_add_gumbel_noise);
    }

    // ========================================================================
    // Finetune parameters
    // ========================================================================
    if (is_set(FLAGS_learning_rate)) {
        params.lr.lr0 = std::stof(FLAGS_learning_rate);
    }

    if (is_set(FLAGS_learning_rate_min)) {
        params.lr.lr_min = std::stof(FLAGS_learning_rate_min);
    }

    if (is_set(FLAGS_learning_rate_decay_epochs)) {
        params.lr.decay_epochs = std::stoi(FLAGS_learning_rate_decay_epochs);
    }

    if (is_set(FLAGS_weight_decay)) {
        params.lr.wd = std::stof(FLAGS_weight_decay);
    }

    if (is_set(FLAGS_val_split)) {
        params.val_split = std::stof(FLAGS_val_split);
    }

    if (FLAGS_epochs != 0) {
        params.lr.epochs = FLAGS_epochs;
    }

    if (is_set(FLAGS_optimizer)) {
        if (FLAGS_optimizer == "adamw") {
            params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        } else if (FLAGS_optimizer == "sgd") {
            params.optimizer = GGML_OPT_OPTIMIZER_TYPE_SGD;
        }
    }

    // ========================================================================
    // Debug parameters
    // ========================================================================
    if (FLAGS_check) {
        params.check = true;
    }

    if (FLAGS_save_logits) {
        params.save_logits = true;
    }

    if (is_set(FLAGS_logits_output_dir) && FLAGS_logits_output_dir != "data") {
        params.logits_output_dir = FLAGS_logits_output_dir;
    }

    if (is_set(FLAGS_tensor_filter)) {
        params.tensor_filter.push_back(FLAGS_tensor_filter);
    }

    // ========================================================================
    // Preset defaults
    // ========================================================================
    if (FLAGS_tts_oute_default) {
        params.model.hf_repo = "lmstudio-community/Qwen2.5-OuteTTS-1.0-0.6B-GGUF";
        params.model.hf_file = "*.gguf";
        params.vocoder.model.hf_repo = "ggml-org/OuteTTS-1.0-0.6B";
    }

    if (FLAGS_embd_gemma_default) {
        params.model.hf_repo = "google/gemma-3-4b-it-qat-q4_k_m-gguf";
        params.model.hf_file = "*q4_k_m.gguf";
        params.port = 8081;
    }

    if (FLAGS_spec_default) {
        params.speculative.ngram_mod.n_match = 2;
        params.speculative.ngram_mod.n_min = 1;
        params.speculative.ngram_mod.n_max = 4;
    }

    // ========================================================================
    // Post-processing (same as original arg.cpp)
    // ========================================================================
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams, &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams_batch, &params.cpuparams_batch);

    // Escape processing
    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
        for (auto & seq_breaker : params.sampling.dry_sequence_breakers) {
            string_process_escapes(seq_breaker);
        }
    }

    // KV overrides terminator
    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    // Pad tensor_buft_overrides
    const size_t ntbo = lhm_max_tensor_buft_overrides();
    while (params.tensor_buft_overrides.size() < ntbo) {
        params.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.speculative.draft.tensor_buft_overrides.empty()) {
        params.speculative.draft.tensor_buft_overrides.push_back({nullptr, nullptr});
    }
}

} // namespace lhm

// ============================================================================
// Model handling (migrated from arg.cpp)
// ============================================================================

static const std::initializer_list<enum lhm_example> mmproj_examples = {
    LHM_EXAMPLE_MTMD,
    LHM_EXAMPLE_SERVER,
    LHM_EXAMPLE_CLI,
};

struct handle_model_result {
    bool found_mmproj = false;
    common_params_model mmproj;

    bool found_mtp = false;
    common_params_model mtp;

    bool found_preset = false;
    std::string preset_path;
};

static handle_model_result common_params_handle_model(struct common_params_model & model,
                                                      const common_download_opts & opts) {
    handle_model_result result;

    if (!model.docker_repo.empty()) {
        model.path = common_docker_resolve_model(model.docker_repo);
        model.name = model.docker_repo;
    } else if (!model.hf_repo.empty()) {
        // If -m was used with -hf, treat the model "path" as the hf_file to download
        if (model.hf_file.empty() && !model.path.empty()) {
            model.hf_file = model.path;
            model.path = "";
            throw std::exception("hf_file must be specified when using -hf with -m");
        }
    } else if (!model.url.empty()) {
        if (model.path.empty()) {
            auto f = string_split<std::string>(model.url, '#').front();
            f = string_split<std::string>(f, '?').front();
            model.path = fs_get_cache_file(string_split<std::string>(f, '/').back());
        }

    }

    return result;
}

bool common_params_handle_models(common_params & params, lhm_example curr_ex) {
    const bool spec_type_draft_mtp = std::find(params.speculative.types.begin(),
                                         params.speculative.types.end(),
                                         COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    common_download_opts opts;
    opts.bearer_token    = params.hf_token;
    opts.offline         = params.offline;
    opts.skip_download   = params.skip_download;
    opts.download_mtp    = spec_type_draft_mtp;
    opts.download_mmproj = !params.no_mmproj && params.mmproj.path.empty() && params.mmproj.url.empty();

    // sub-models (draft, mmproj, vocoder) are explicitly specified by the user,
    // so we should not auto-discover mtp/mmproj siblings for them
    common_download_opts sub_opts = opts;
    sub_opts.download_mtp    = false;
    sub_opts.download_mmproj = false;

    try {
        auto res = common_params_handle_model(params.model, opts);
        if (res.found_preset) {
            if (!params.models_preset.empty()) {
                throw std::invalid_argument("cannot use both --models-preset and -hf with a preset.ini file");
            }
            // if HF repo is a preset repo, we simply run server in router mode with the preset.ini file
            params.models_preset_hf = params.model.hf_repo; // only for showing a warning
            params.models_preset    = res.preset_path;
            params.model = common_params_model{}; // make sure to clear model, so server starts in router mode
            return true;
        }

        if (params.no_mmproj) {
            params.mmproj = {};
        } else if (res.found_mmproj && params.mmproj.path.empty() && params.mmproj.url.empty()) {
            // optionally, handle mmproj model when -hf is specified
            params.mmproj = res.mmproj;
        }
        // only download mmproj if the current example is using it
        for (const auto & ex : mmproj_examples) {
            if (curr_ex == ex) {
                common_params_handle_model(params.mmproj, sub_opts);
                break;
            }
        }

        // when --spec-type mtp is set and no draft model was provided explicitly,
        // fall back to the MTP head discovered alongside the -hf model
        if (spec_type_draft_mtp && res.found_mtp &&
            params.speculative.draft.mparams.path.empty() &&
            params.speculative.draft.mparams.hf_repo.empty() &&
            params.speculative.draft.mparams.url.empty()) {
            params.speculative.draft.mparams.path = res.mtp.path;
        }
        common_params_handle_model(params.speculative.draft.mparams, sub_opts);
        common_params_handle_model(params.vocoder.model,             sub_opts);
        return true;
    } catch (const common_skip_download_exception &) {
        return false;
    } catch (const std::exception &) {
        throw;
    }
}
