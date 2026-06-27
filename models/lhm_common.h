// Various helper functions and utilities

#pragma once

#include "lassert.h"
#include "lhm_cpp.h"

#include "ggml-opt.h"
#include "ggml.h"

#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <algorithm>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

bool string_parse_kv_override(const char * data, std::vector<lhm_model_kv_override> & overrides);

void common_params_print_info(const common_params & params, bool print_devices = true);
std::string common_params_get_system_info(const common_params & params);

std::string string_from(const struct lhm_context * ctx, const std::vector<lhm_token> & tokens);
std::string string_from(const struct lhm_context * ctx, const struct lhm_batch & batch);

//
// Model utils
//

struct common_sampler;

// note: defines the model, context, samplers, ets. lifetimes
struct common_init_result {
    common_init_result(common_params & params, bool model_only = false);
    ~common_init_result();

    lhm_model * model();
    lhm_context * context();

    common_sampler * sampler(lhm_seq_id seq_id);
    void reset_samplers();

    std::vector<lhm_adapter_lora_ptr> & lora();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_init_result_ptr = std::unique_ptr<common_init_result>;

common_init_result_ptr common_init_from_params(common_params & params, bool model_only = false);

struct lhm_model_params     common_model_params_to_llama  (      common_params & params);
struct lhm_context_params   common_context_params_to_llama(const common_params & params);
struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params);

// clear LoRA adapters from context, then apply new list of adapters
void common_set_adapter_lora(struct lhm_context * ctx, std::vector<common_adapter_lora_info> & lora);

// model endpoint from env
std::string common_get_model_endpoint();

//
// Context utils
//

enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO           = 0, // seq_rm not supported (e.g. no memory module)
    COMMON_CONTEXT_SEQ_RM_TYPE_PART         = 1, // can seq_rm partial sequences
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL         = 2, // can seq_rm full sequences only
    COMMON_CONTEXT_SEQ_RM_TYPE_RS = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};

// check if the lhm_context can remove sequences
// note: clears the memory of the context
common_context_seq_rm_type common_context_can_seq_rm(lhm_context * ctx);

// aborts execution on failure
void common_context_seq_rm (lhm_context * ctx, lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1);
void common_context_seq_add(lhm_context * ctx, lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1, lhm_pos delta);
void common_context_seq_cp (lhm_context * ctx, lhm_seq_id seq_id_src, lhm_seq_id seq_id_dst, lhm_pos p0, lhm_pos p1);

//
// Batch utils
//

void common_batch_clear(struct lhm_batch & batch);

void common_batch_add(
                 struct lhm_batch & batch,
                        lhm_token   id,
                          lhm_pos   pos,
    const std::vector<lhm_seq_id> & seq_ids,
                               bool   logits);

// decodes a single batch of tokens for a prompt and manages session tokens
//
// Note: We save state before the last token so that we can replay it to ensure
// compatibility with all memory types. Recurrent/hybrid models cannot remove
// tokens from memory, so this approach works across all model architectures.
bool common_prompt_batch_decode(
              struct lhm_context * ctx,
    const std::vector<lhm_token> & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state);

// replays the last token after loading state to regenerate logits
// used after loading session state to ensure the sampling context has valid logits
bool common_replay_last_token(struct lhm_context * ctx, lhm_token last_token, int32_t pos);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<lhm_token> common_tokenize(
  const struct lhm_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<lhm_token> common_tokenize(
    const struct lhm_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string common_token_to_piece(
        const struct lhm_context * ctx,
                       lhm_token   token,
                       bool          special = true);

std::string common_token_to_piece(
          const struct lhm_vocab * vocab,
                       lhm_token   token,
                       bool          special = true);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string common_detokenize(
            const struct lhm_context * ctx,
        const std::vector<lhm_token> & tokens,
                                  bool   special = true);

std::string common_detokenize(
              const struct lhm_vocab * vocab,
        const std::vector<lhm_token> & tokens,
                                  bool   special = true);

//
// Embedding utils
//

// TODO: replace embd_norm with an enum
void common_embd_normalize(const float * inp, float * out, int n, int embd_norm);

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utils
//

struct common_control_vector_data {
    int n_embd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
    std::vector<float> data;
};

// Load control vectors, scale each by strength, and add them together.
// On error, returns {-1, empty}
common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos);

//
// Split utils
//

namespace {

const char * const LLM_KV_SPLIT_NO            = "split.no";
const char * const LLM_KV_SPLIT_COUNT         = "split.count";
const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

}

//
// MoE utils
//

const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

inline std::string llm_ffn_exps_block_regex(int idx) {
    return string_format("blk\\.%d%s", idx, LLM_FFN_EXPS_REGEX);
}

inline lhm_model_tensor_buft_override llm_ffn_exps_cpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() };
}

//
// training utils
//

ggml_opt_dataset_t common_opt_dataset_init(struct lhm_context * ctx, const std::vector<lhm_token> & tokens, int64_t stride);

// "adamw" or "sgd" (case insensitive)
enum ggml_opt_optimizer_type common_opt_get_optimizer(const char *);

//
// prompt utils
//

struct common_prompt_checkpoint {
    int64_t n_tokens;

    lhm_pos pos_min;
    lhm_pos pos_max;

    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;

    size_t size() const;

    bool empty() const;
    void clear();

    void update_pos(
            int64_t n_tokens,
            lhm_pos pos_min,
            lhm_pos pos_max);

    void update_tgt(
            lhm_context * ctx,
            lhm_seq_id seq_id,
            lhm_state_seq_flags flags);

    void update_dft(
            lhm_context * ctx,
            lhm_seq_id seq_id,
            lhm_state_seq_flags flags);

    void load_tgt(
            lhm_context * ctx,
            lhm_seq_id seq_id,
            lhm_state_seq_flags flags) const;

    void load_dft(
            lhm_context * ctx,
            lhm_seq_id seq_id,
            lhm_state_seq_flags flags) const;

    void clear_tgt();
    void clear_dft();
};
