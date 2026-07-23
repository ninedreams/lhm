#pragma once

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-opt.h>
#include <gguf.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LHM_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LHM_BUILD
#            define __declspec(dllexport)
#        else
#            define __declspec(dllimport)
#        endif
#    else
#        define __attribute__ ((visibility ("default")))
#    endif
#else
#    define LHM_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LHM_DEFAULT_SEED 0xFFFFFFFF

#define LHM_TOKEN_NULL -1

#define LHM_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LHM_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LHM_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LHM_SESSION_MAGIC   LHM_FILE_MAGIC_GGSN
#define LHM_SESSION_VERSION 9

#define LHM_STATE_SEQ_MAGIC   LHM_FILE_MAGIC_GGSQ
#define LHM_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct lhm_vocab;
    struct lhm_model;
    struct lhm_context;
    struct lhm_sampler;

    typedef struct lhm_memory_i * lhm_memory_t;

    typedef int32_t lhm_pos;
    typedef int32_t lhm_token;
    typedef int32_t lhm_seq_id;

    enum lhm_vocab_type {
        LHM_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LHM_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LHM_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LHM_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LHM_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LHM_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LHM_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    enum lhm_rope_type {
        LHM_ROPE_TYPE_NONE   = -1,
        LHM_ROPE_TYPE_NORM   = 0,
        LHM_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LHM_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LHM_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LHM_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum lhm_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LHM_TOKEN_TYPE_UNDEFINED    = 0,
        LHM_TOKEN_TYPE_NORMAL       = 1,
        LHM_TOKEN_TYPE_UNKNOWN      = 2,
        LHM_TOKEN_TYPE_CONTROL      = 3,
        LHM_TOKEN_TYPE_USER_DEFINED = 4,
        LHM_TOKEN_TYPE_UNUSED       = 5,
        LHM_TOKEN_TYPE_BYTE         = 6,
    };

    enum lhm_token_attr {
        LHM_TOKEN_ATTR_UNDEFINED    = 0,
        LHM_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LHM_TOKEN_ATTR_UNUSED       = 1 << 1,
        LHM_TOKEN_ATTR_NORMAL       = 1 << 2,
        LHM_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LHM_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LHM_TOKEN_ATTR_BYTE         = 1 << 5,
        LHM_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LHM_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LHM_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LHM_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum lhm_ftype {
        LHM_FTYPE_ALL_F32              = 0,
        LHM_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LHM_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LHM_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LHM_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LHM_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LHM_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LHM_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LHM_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LHM_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LHM_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LHM_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LHM_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LHM_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LHM_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LHM_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LHM_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LHM_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        LHM_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
        LHM_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors
        LHM_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors

        LHM_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum lhm_rope_scaling_type {
        LHM_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LHM_ROPE_SCALING_TYPE_NONE        = 0,
        LHM_ROPE_SCALING_TYPE_LINEAR      = 1,
        LHM_ROPE_SCALING_TYPE_YARN        = 2,
        LHM_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LHM_ROPE_SCALING_TYPE_MAX_VALUE   = LHM_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum lhm_pooling_type {
        LHM_POOLING_TYPE_UNSPECIFIED = -1,
        LHM_POOLING_TYPE_NONE = 0,
        LHM_POOLING_TYPE_MEAN = 1,
        LHM_POOLING_TYPE_CLS  = 2,
        LHM_POOLING_TYPE_LAST = 3,
        LHM_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum lhm_attention_type {
        LHM_ATTENTION_TYPE_UNSPECIFIED = -1,
        LHM_ATTENTION_TYPE_CAUSAL      = 0,
        LHM_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum lhm_flash_attn_type {
        LHM_FLASH_ATTN_TYPE_AUTO     = -1,
        LHM_FLASH_ATTN_TYPE_DISABLED = 0,
        LHM_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    const char * lhm_flash_attn_type_name(enum lhm_flash_attn_type flash_attn_type);

    enum lhm_split_mode {
        LHM_SPLIT_MODE_NONE   = 0, // single GPU
        LHM_SPLIT_MODE_LAYER  = 1, // split layers and KV across GPUs
        LHM_SPLIT_MODE_ROW    = 2, // split layers and KV across GPUs, use tensor parallelism if supported
        LHM_SPLIT_MODE_TENSOR = 3,
    };

    enum lhm_context_type {
        LHM_CONTEXT_TYPE_DEFAULT = 0,
        LHM_CONTEXT_TYPE_MTP     = 1,
    };

    typedef struct lhm_token_data {
        lhm_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } lhm_token_data;

    typedef struct lhm_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        lhm_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } lhm_token_data_array;

    typedef bool (*lhm_progress_callback)(float progress, void * user_data);

    // Input data for lhm_encode/lhm_decode
    // A lhm_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by lhm_encode/lhm_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct lhm_batch {
        int32_t n_tokens;

        lhm_token  *  token;
        float        *  embd;
        lhm_pos    *  pos;
        int32_t      *  n_seq_id;
        lhm_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } lhm_batch;

    enum lhm_model_kv_override_type {
        LHM_KV_OVERRIDE_TYPE_INT,
        LHM_KV_OVERRIDE_TYPE_FLOAT,
        LHM_KV_OVERRIDE_TYPE_BOOL,
        LHM_KV_OVERRIDE_TYPE_STR,
    };

    enum lhm_model_meta_key {
        LHM_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LHM_MODEL_META_KEY_SAMPLING_TOP_K,
        LHM_MODEL_META_KEY_SAMPLING_TOP_P,
        LHM_MODEL_META_KEY_SAMPLING_MIN_P,
        LHM_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LHM_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LHM_MODEL_META_KEY_SAMPLING_TEMP,
        LHM_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LHM_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LHM_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LHM_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LHM_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    struct lhm_model_kv_override {
        enum lhm_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct lhm_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct lhm_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct lhm_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum lhm_split_mode split_mode; // how to split the model across multiple GPUs

        // the GPU that is used for the entire model when split_mode is LHM_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: lhm_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        lhm_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct lhm_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool use_mmap;        // use mmap if possible
        bool use_direct_io;   // use direct io, takes precedence over use_mmap when supported
        bool use_mlock;       // force system to keep model in RAM
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
    };

    struct lhm_sampler_seq_config {
        lhm_seq_id           seq_id;
        struct lhm_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    struct lhm_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to lhm_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_rs_seq;          // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
        uint32_t n_outputs_max;     // max outputs in a ubatch (0 = n_batch)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum lhm_context_type      ctx_type;          // set the context type (e.g. MTP)
        enum lhm_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum lhm_rope_scaling_type`
        enum lhm_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum lhm_attention_type    attention_type;    // attention type to use for embeddings
        enum lhm_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of lhm_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix

        // [EXPERIMENTAL]
        // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
        // note: the samplers must be sampler chains (i.e. use lhm_sampler_chain_init)
        struct lhm_sampler_seq_config * samplers;
        size_t                            n_samplers;

        // a source/target/parent context
        // can be utilized in various ways, for example by sharing results or lhm_memory between 2 contexts
        struct lhm_context * ctx_other;
    };

    struct lhm_model_tensor_override {
        const char * pattern;
        enum ggml_type type;
    };

    struct lhm_model_imatrix_data {
        const char * name;
        const float * data;
        size_t size;
    };

    // model quantization parameters
    typedef struct lhm_model_quantize_params {
        int32_t nthread;                                            // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum lhm_ftype ftype;                                     // quantize to this lhm_ftype
        enum ggml_type output_tensor_type;                          // output tensor type
        enum ggml_type token_embedding_type;                        // token embeddings tensor type
        bool allow_requantize;                                      // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;                                // quantize output.weight
        bool only_copy;                                             // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                                                  // quantize all tensors to the default type
        bool keep_split;                                            // quantize to the same number of shards
        bool dry_run;                                               // calculate and show the final quantization size without performing quantization
        const struct lhm_model_imatrix_data * imatrix;            // pointer to importance matrix data
        const struct lhm_model_kv_override * kv_overrides;        // pointer to kv overrides
        const struct lhm_model_tensor_override * tt_overrides;    // pointer to tensor overrides
        const int32_t * prune_layers;                               // pointer to layer indices to prune
    } lhm_model_quantize_params;

    typedef struct lhm_logit_bias {
        lhm_token token;
        float bias;
    } lhm_logit_bias;

    typedef struct lhm_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } lhm_sampler_chain_params;

    // used in chat template
    typedef struct lhm_chat_message {
        const char * role;
        const char * content;
    } lhm_chat_message;

    // lora adapter
    struct lhm_adapter_lora;

    // Helpers for getting default parameters
    struct lhm_model_params          lhm_model_default_params(void);
    struct lhm_context_params        lhm_context_default_params(void);
    struct lhm_sampler_chain_params  lhm_sampler_chain_default_params(void);
    struct lhm_model_quantize_params lhm_model_quantize_default_params(void);

    // Initialize the lhm + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    void lhm_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    void lhm_backend_free(void);

    //optional:
    void lhm_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    void lhm_attach_threadpool(
            struct lhm_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    void lhm_detach_threadpool(struct lhm_context * ctx);

    typedef void (*lhm_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    // Create a new model from GGUF metadata as well as a function to set the tensor data
    //   - tensors are created as GGML_TYPE_F32 by default,
    //     override by adding a tensor with the same name but a different name to the context
    struct lhm_model * lhm_model_init_from_user(
                    struct gguf_context * metadata,
          lhm_model_set_tensor_data_t   set_tensor_data,    // function to initialize tensor data with
                                   void * set_tensor_data_ud, // userdata for function
              struct lhm_model_params   params);

    // Load a model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use lhm_model_load_from_splits
    struct lhm_model * lhm_model_load_from_file(
                             const char * path_model,
              struct lhm_model_params   params);

    // Load a model from an open FILE pointer
    struct lhm_model * lhm_model_load_from_file_ptr(
                                   FILE * file,
              struct lhm_model_params   params);

    // Load a model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    struct lhm_model * lhm_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct lhm_model_params    params);

    void lhm_model_free(struct lhm_model * model);

    struct lhm_context * lhm_init_from_model(
                     struct lhm_model * model,
            struct lhm_context_params   params);

    // Frees all allocated memory
    void lhm_free(struct lhm_context * ctx);

    int64_t lhm_time_us(void);

    size_t lhm_max_devices(void);
    size_t lhm_max_parallel_sequences(void);
    size_t lhm_max_tensor_buft_overrides(void);

    bool lhm_supports_mmap       (void);
    bool lhm_supports_mlock      (void);
    bool lhm_supports_gpu_offload(void);
    bool lhm_supports_rpc        (void);

    // NOTE: After creating a lhm_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via lhm_context_params may differ from the actual values used by the context
    uint32_t lhm_n_ctx      (const struct lhm_context * ctx);
    uint32_t lhm_n_ctx_seq  (const struct lhm_context * ctx);
    uint32_t lhm_n_batch    (const struct lhm_context * ctx);
    uint32_t lhm_n_ubatch   (const struct lhm_context * ctx);
    uint32_t lhm_n_seq_max  (const struct lhm_context * ctx);
    uint32_t lhm_n_rs_seq   (const struct lhm_context * ctx);

    DEPRECATED(int32_t lhm_n_ctx_train(const struct lhm_model * model), "use lhm_model_n_ctx_train instead");
    DEPRECATED(int32_t lhm_n_embd     (const struct lhm_model * model), "use lhm_model_n_embd instead");
    DEPRECATED(int32_t lhm_n_layer    (const struct lhm_model * model), "use lhm_model_n_layer instead");
    DEPRECATED(int32_t lhm_n_head     (const struct lhm_model * model), "use lhm_model_n_head instead");

    DEPRECATED(int32_t lhm_n_vocab    (const struct lhm_vocab * vocab), "use lhm_vocab_n_tokens instead");

    const struct lhm_model * lhm_get_model   (const struct lhm_context * ctx);
              lhm_memory_t   lhm_get_memory  (const struct lhm_context * ctx);
     enum lhm_pooling_type   lhm_pooling_type(const struct lhm_context * ctx); // TODO: rename to lhm_get_pooling_type

    const struct lhm_vocab * lhm_model_get_vocab(const struct lhm_model * model);
    enum lhm_rope_type       lhm_model_rope_type(const struct lhm_model * model);

    int32_t lhm_model_n_ctx_train(const struct lhm_model * model);
    int32_t lhm_model_n_embd     (const struct lhm_model * model);
    int32_t lhm_model_n_embd_inp (const struct lhm_model * model);
    int32_t lhm_model_n_embd_out (const struct lhm_model * model);
    int32_t lhm_model_n_layer    (const struct lhm_model * model);
    int32_t lhm_model_n_layer_nextn(const struct lhm_model * model);
    int32_t lhm_model_n_head     (const struct lhm_model * model);
    int32_t lhm_model_n_head_kv  (const struct lhm_model * model);
    int32_t lhm_model_n_swa      (const struct lhm_model * model);

    // Get the model's RoPE frequency scaling factor
    float lhm_model_rope_freq_scale_train(const struct lhm_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    uint32_t lhm_model_n_cls_out(const struct lhm_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    const char * lhm_model_cls_label(const struct lhm_model * model, uint32_t i);

    enum lhm_vocab_type lhm_vocab_type(const struct lhm_vocab * vocab);

    int32_t lhm_vocab_n_tokens(const struct lhm_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    int32_t lhm_model_meta_val_str(const struct lhm_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    int32_t lhm_model_meta_count(const struct lhm_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    const char * lhm_model_meta_key_str(enum lhm_model_meta_key key);

    // Get metadata key name by index
    int32_t lhm_model_meta_key_by_index(const struct lhm_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    int32_t lhm_model_meta_val_str_by_index(const struct lhm_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    int32_t lhm_model_desc(const struct lhm_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    uint64_t lhm_model_size(const struct lhm_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    const char * lhm_model_chat_template(const struct lhm_model * model, const char * name);

    // Returns the total number of parameters in the model
    uint64_t lhm_model_n_params(const struct lhm_model * model);

    // Returns true if the model contains an encoder that requires lhm_encode() call
    bool lhm_model_has_encoder(const struct lhm_model * model);

    // Returns true if the model contains a decoder that requires lhm_decode() call
    bool lhm_model_has_decoder(const struct lhm_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    lhm_token lhm_model_decoder_start_token(const struct lhm_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    bool lhm_model_is_recurrent(const struct lhm_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    bool lhm_model_is_hybrid(const struct lhm_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    bool lhm_model_is_diffusion(const struct lhm_model * model);

    // Returns 0 on success
    uint32_t lhm_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const lhm_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    struct lhm_adapter_lora * lhm_adapter_lora_init(
            struct lhm_model * model,
            const char * path_lora);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    int32_t lhm_adapter_meta_val_str(const struct lhm_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    int32_t lhm_adapter_meta_count(const struct lhm_adapter_lora * adapter);

    // Get metadata key name by index
    int32_t lhm_adapter_meta_key_by_index(const struct lhm_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    int32_t lhm_adapter_meta_val_str_by_index(const struct lhm_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
    void lhm_adapter_lora_free(struct lhm_adapter_lora * adapter);

    // Get the invocation tokens if the current lora is an alora
    uint64_t            lhm_adapter_get_alora_n_invocation_tokens(const struct lhm_adapter_lora * adapter);
    const lhm_token * lhm_adapter_get_alora_invocation_tokens  (const struct lhm_adapter_lora * adapter);

    // The following functions operate on a lhm_context, hence the naming: lhm_verb_...

    // Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
    int32_t lhm_set_adapters_lora(
            struct lhm_context * ctx,
            struct lhm_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Apply a loaded control vector to a lhm_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See lhm_control_vector_load in common to load a control vector.
    int32_t lhm_set_adapter_cvec(
            struct lhm_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    void lhm_memory_clear(
            lhm_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    bool lhm_memory_seq_rm(
            lhm_memory_t mem,
              lhm_seq_id seq_id,
                 lhm_pos p0,
                 lhm_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    void lhm_memory_seq_cp(
            lhm_memory_t mem,
              lhm_seq_id seq_id_src,
              lhm_seq_id seq_id_dst,
                 lhm_pos p0,
                 lhm_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    void lhm_memory_seq_keep(
            lhm_memory_t mem,
              lhm_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    void lhm_memory_seq_add(
            lhm_memory_t mem,
              lhm_seq_id seq_id,
                 lhm_pos p0,
                 lhm_pos p1,
                 lhm_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    void lhm_memory_seq_div(
            lhm_memory_t mem,
              lhm_seq_id seq_id,
                 lhm_pos p0,
                 lhm_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    lhm_pos lhm_memory_seq_pos_min(
            lhm_memory_t mem,
              lhm_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    lhm_pos lhm_memory_seq_pos_max(
            lhm_memory_t mem,
              lhm_seq_id seq_id);

    // Check if the memory supports shifting
    bool lhm_memory_can_shift(lhm_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    size_t lhm_state_get_size(struct lhm_context * ctx);
    DEPRECATED(size_t lhm_get_state_size(struct lhm_context * ctx),
        "use lhm_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    size_t lhm_state_get_data(
            struct lhm_context * ctx,
                         uint8_t * dst,
                          size_t   size);

    // Set the state reading from the specified address
    // Returns the number of bytes read
    size_t lhm_state_set_data(
            struct lhm_context * ctx,
                   const uint8_t * src,
                          size_t   size);

    // Save/load session file
    bool lhm_state_load_file(
            struct lhm_context * ctx,
                      const char * path_session,
                     lhm_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

    bool lhm_state_save_file(
            struct lhm_context * ctx,
                      const char * path_session,
               const lhm_token * tokens,
                          size_t   n_token_count);
    DEPRECATED(bool lhm_save_session_file(
            struct lhm_context * ctx,
                      const char * path_session,
               const lhm_token * tokens,
                          size_t   n_token_count),
        "use lhm_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    size_t lhm_state_seq_get_size(
            struct lhm_context * ctx,
                    lhm_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    size_t lhm_state_seq_get_data(
            struct lhm_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    lhm_seq_id   seq_id);

    // Copy the sequence data (originally copied with `lhm_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    size_t lhm_state_seq_set_data(
            struct lhm_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    lhm_seq_id   dest_seq_id);

    size_t lhm_state_seq_save_file(
            struct lhm_context * ctx,
                      const char * filepath,
                    lhm_seq_id   seq_id,
               const lhm_token * tokens,
                          size_t   n_token_count);

    size_t lhm_state_seq_load_file(
            struct lhm_context * ctx,
                      const char * filepath,
                    lhm_seq_id   dest_seq_id,
                     lhm_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

#define LHM_STATE_SEQ_FLAGS_NONE 0

// for backwards-compat
#define LHM_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LHM_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define LHM_STATE_SEQ_FLAGS_ON_DEVICE 2

    typedef uint32_t lhm_state_seq_flags;

    size_t lhm_state_seq_get_size_ext(
            struct lhm_context * ctx,
                    lhm_seq_id   seq_id,
           lhm_state_seq_flags   flags);

    size_t lhm_state_seq_get_data_ext(
            struct lhm_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    lhm_seq_id   seq_id,
           lhm_state_seq_flags   flags);

    size_t lhm_state_seq_set_data_ext(
            struct lhm_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    lhm_seq_id   dest_seq_id,
           lhm_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by lhm_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    struct lhm_batch lhm_batch_get_one(
                  lhm_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with lhm_batch_free()
    // If embd != 0, lhm_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, lhm_batch.token will be allocated to store n_tokens lhm_token
    // The rest of the lhm_batch members are allocated with size n_tokens
    // All members are left uninitialized
    struct lhm_batch lhm_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with lhm_batch_init()
    void lhm_batch_free(struct lhm_batch batch);

    // Process a batch of tokens.
    // In contrast to lhm_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    int32_t lhm_encode(
            struct lhm_context * ctx,
              struct lhm_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using lhm_memory_seq_pos_min() and lhm_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    int32_t lhm_decode(
            struct lhm_context * ctx,
              struct lhm_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    void lhm_set_n_threads(struct lhm_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    int32_t lhm_n_threads(struct lhm_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    int32_t lhm_n_threads_batch(struct lhm_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with lhm_get_embeddings()
    void lhm_set_embeddings(struct lhm_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    void lhm_set_causal_attn(struct lhm_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during lhm_decode() to load and cache their weights.
    //
    // note: using this can cause extra graph reallocations because it changes the graph topology with MoE models,
    //       so it is generally not recommended to use in practice. will be removed in the future
    DEPRECATED(void lhm_set_warmup(struct lhm_context * ctx, bool warmup),
            "user code should do warmup runs manually [TAG_LHM_GRAPH_NO_WARMUP]");

    // Set abort callback
    void lhm_set_abort_callback(struct lhm_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    void lhm_synchronize(struct lhm_context * ctx);

    // Token logits obtained from the last call to lhm_decode()
    // The logits for which lhm_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which lhm_batch.logits[i] != 0
    // Cols: n_vocab
    float * lhm_get_logits(struct lhm_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // lhm_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indices can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    float * lhm_get_logits_ith(struct lhm_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LHM_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which lhm_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    float * lhm_get_embeddings(struct lhm_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // lhm_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    float * lhm_get_embeddings_ith(struct lhm_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LHM_POOLING_TYPE_NONE
    // when pooling_type == LHM_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    float * lhm_get_embeddings_seq(struct lhm_context * ctx, lhm_seq_id seq_id);

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the lhm_context was created with at least one lhm_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // Returns LHM_TOKEN_NULL if no token was sampled.
    lhm_token lhm_get_sampled_token_ith(struct lhm_context * ctx, int32_t i);

    // Get the backend sampled probabilities for the ith token
    // The index matches lhm_get_sampled_token_ith().
    // Returns NULL if no probabilities were generated.
    float *  lhm_get_sampled_probs_ith      (struct lhm_context * ctx, int32_t i);
    uint32_t lhm_get_sampled_probs_count_ith(struct lhm_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    float *  lhm_get_sampled_logits_ith      (struct lhm_context * ctx, int32_t i);
    uint32_t lhm_get_sampled_logits_count_ith(struct lhm_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    lhm_token * lhm_get_sampled_candidates_ith      (struct lhm_context * ctx, int32_t i);
    uint32_t      lhm_get_sampled_candidates_count_ith(struct lhm_context * ctx, int32_t i);

    //
    // Vocab
    //

    const char * lhm_vocab_get_text(const struct lhm_vocab * vocab, lhm_token token);

    float lhm_vocab_get_score(const struct lhm_vocab * vocab, lhm_token token);

    enum lhm_token_attr lhm_vocab_get_attr(const struct lhm_vocab * vocab, lhm_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    bool lhm_vocab_is_eog(const struct lhm_vocab * vocab, lhm_token token);

    // Identify if Token Id is a control token or a render-able token
    bool lhm_vocab_is_control(const struct lhm_vocab * vocab, lhm_token token);

    // Special tokens
    lhm_token lhm_vocab_bos(const struct lhm_vocab * vocab); // beginning-of-sentence
    lhm_token lhm_vocab_eos(const struct lhm_vocab * vocab); // end-of-sentence
    lhm_token lhm_vocab_eot(const struct lhm_vocab * vocab); // end-of-turn
    lhm_token lhm_vocab_sep(const struct lhm_vocab * vocab); // sentence separator
    lhm_token lhm_vocab_nl (const struct lhm_vocab * vocab); // next-line
    lhm_token lhm_vocab_pad(const struct lhm_vocab * vocab); // padding
    lhm_token lhm_vocab_mask(const struct lhm_vocab * vocab); // mask

    bool lhm_vocab_get_add_bos(const struct lhm_vocab * vocab);
    bool lhm_vocab_get_add_eos(const struct lhm_vocab * vocab);
    bool lhm_vocab_get_add_sep(const struct lhm_vocab * vocab);

    lhm_token lhm_vocab_fim_pre(const struct lhm_vocab * vocab);
    lhm_token lhm_vocab_fim_suf(const struct lhm_vocab * vocab);
    lhm_token lhm_vocab_fim_mid(const struct lhm_vocab * vocab);
    lhm_token lhm_vocab_fim_pad(const struct lhm_vocab * vocab);
    lhm_token lhm_vocab_fim_rep(const struct lhm_vocab * vocab);
    lhm_token lhm_vocab_fim_sep(const struct lhm_vocab * vocab);

    DEPRECATED(const char * lhm_token_get_text(const struct lhm_vocab * vocab, lhm_token token), "use lhm_vocab_get_text instead");
    DEPRECATED(float lhm_token_get_score(const struct lhm_vocab * vocab, lhm_token token), "use lhm_vocab_get_score instead");
    DEPRECATED(enum lhm_token_attr lhm_token_get_attr(const struct lhm_vocab * vocab, lhm_token token), "use lhm_vocab_get_attr instead");
    DEPRECATED(bool lhm_token_is_eog(const struct lhm_vocab * vocab, lhm_token token), "use lhm_vocab_is_eog instead");
    DEPRECATED(bool lhm_token_is_control(const struct lhm_vocab * vocab, lhm_token token), "use lhm_vocab_is_control instead");
    DEPRECATED(lhm_token lhm_token_bos(const struct lhm_vocab * vocab), "use lhm_vocab_bos instead");
    DEPRECATED(lhm_token lhm_token_eos(const struct lhm_vocab * vocab), "use lhm_vocab_eos instead");
    DEPRECATED(lhm_token lhm_token_eot(const struct lhm_vocab * vocab), "use lhm_vocab_eot instead");
    DEPRECATED(lhm_token lhm_token_cls(const struct lhm_vocab * vocab), "use lhm_vocab_cls instead");
    DEPRECATED(lhm_token lhm_token_sep(const struct lhm_vocab * vocab), "use lhm_vocab_sep instead");
    DEPRECATED(lhm_token lhm_token_nl (const struct lhm_vocab * vocab), "use lhm_vocab_nl instead");
    DEPRECATED(lhm_token lhm_token_pad(const struct lhm_vocab * vocab), "use lhm_vocab_pad instead");
    DEPRECATED(bool lhm_add_bos_token(const struct lhm_vocab * vocab), "use lhm_vocab_get_add_bos instead");
    DEPRECATED(bool lhm_add_eos_token(const struct lhm_vocab * vocab), "use lhm_vocab_get_add_eos instead");
    DEPRECATED(lhm_token lhm_token_fim_pre(const struct lhm_vocab * vocab), "use lhm_vocab_fim_pre instead");
    DEPRECATED(lhm_token lhm_token_fim_suf(const struct lhm_vocab * vocab), "use lhm_vocab_fim_suf instead");
    DEPRECATED(lhm_token lhm_token_fim_mid(const struct lhm_vocab * vocab), "use lhm_vocab_fim_mid instead");
    DEPRECATED(lhm_token lhm_token_fim_pad(const struct lhm_vocab * vocab), "use lhm_vocab_fim_pad instead");
    DEPRECATED(lhm_token lhm_token_fim_rep(const struct lhm_vocab * vocab), "use lhm_vocab_fim_rep instead");
    DEPRECATED(lhm_token lhm_token_fim_sep(const struct lhm_vocab * vocab), "use lhm_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(lhm_token lhm_vocab_cls(const struct lhm_vocab * vocab), // classification
            "use lhm_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    int32_t lhm_tokenize(
        const struct lhm_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     lhm_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    int32_t lhm_token_to_piece(
              const struct lhm_vocab * vocab,
                           lhm_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of lhm_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    int32_t lhm_detokenize(
        const struct lhm_vocab * vocab,
               const lhm_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template.
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple lhm_chat_message
    /// @param n_msg Number of lhm_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    int32_t lhm_chat_apply_template(
                            const char * tmpl,
       const struct lhm_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    int32_t lhm_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = lhm_sampler_chain_default_params();
    //
    //    lhm_sampler * smpl = lhm_sampler_chain_init(sparams);
    //
    //    lhm_sampler_chain_add(smpl, lhm_sampler_init_top_k(50));
    //    lhm_sampler_chain_add(smpl, lhm_sampler_init_top_p(0.9, 1));
    //    lhm_sampler_chain_add(smpl, lhm_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    lhm_sampler_chain_add(smpl, lhm_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        lhm_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const lhm_token id = lhm_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    lhm_sampler_free(smpl);
    //

    typedef void * lhm_sampler_context_t;

    struct lhm_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom lhm_sampler
    struct lhm_sampler_i {
        const char *           (*name)  (const struct lhm_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct lhm_sampler * smpl, lhm_token token);              // can be NULL
        void                   (*apply) (      struct lhm_sampler * smpl, lhm_token_data_array * cur_p); // required
        void                   (*reset) (      struct lhm_sampler * smpl);                                 // can be NULL
        struct lhm_sampler * (*clone) (const struct lhm_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct lhm_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler
        // note: call once per sampler
        bool (*backend_init)(struct lhm_sampler * smpl, ggml_backend_buffer_type_t buft);

        // call after .backend_apply()
        void (*backend_accept)(
                struct lhm_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct lhm_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct lhm_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct lhm_sampler * smpl);
    };

    struct lhm_sampler {
        struct lhm_sampler_i * iface;

        lhm_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with lhm_context_params.samplers when possible
    bool lhm_set_sampler(struct lhm_context * ctx, lhm_seq_id seq_id, struct lhm_sampler * smpl);

    // mirror of lhm_sampler_i:
    struct lhm_sampler * lhm_sampler_init  (      struct lhm_sampler_i * iface, lhm_sampler_context_t ctx);
    const char *           lhm_sampler_name  (const struct lhm_sampler * smpl);
    void                   lhm_sampler_accept(      struct lhm_sampler * smpl, lhm_token token);
    void                   lhm_sampler_apply (      struct lhm_sampler * smpl, lhm_token_data_array * cur_p);
    void                   lhm_sampler_reset (      struct lhm_sampler * smpl);
    struct lhm_sampler * lhm_sampler_clone (const struct lhm_sampler * smpl);
    // important: do not free if the sampler has been added to a lhm_sampler_chain (via lhm_sampler_chain_add)
    void                   lhm_sampler_free  (      struct lhm_sampler * smpl);

    // lhm_sampler_chain
    // a type of lhm_sampler that can chain multiple samplers one after another

    struct lhm_sampler * lhm_sampler_chain_init(struct lhm_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when lhm_sampler_free is called
    void                   lhm_sampler_chain_add(      struct lhm_sampler * chain, struct lhm_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a lhm_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    struct lhm_sampler * lhm_sampler_chain_get(      struct lhm_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    int                    lhm_sampler_chain_n  (const struct lhm_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    struct lhm_sampler * lhm_sampler_chain_remove(   struct lhm_sampler * chain, int32_t i);

    // available samplers:

    struct lhm_sampler * lhm_sampler_init_greedy(void);

    /// seed == LHM_DEFAULT_SEED to use a random seed.
    struct lhm_sampler * lhm_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    struct lhm_sampler * lhm_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    struct lhm_sampler * lhm_sampler_init_top_p      (float   p, size_t min_keep);

    struct lhm_sampler * lhm_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    struct lhm_sampler * lhm_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    struct lhm_sampler * lhm_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    struct lhm_sampler * lhm_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    struct lhm_sampler * lhm_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    struct lhm_sampler * lhm_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `lhm_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    struct lhm_sampler * lhm_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `lhm_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    struct lhm_sampler * lhm_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Initializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    struct lhm_sampler * lhm_sampler_init_grammar(
            const struct lhm_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    struct lhm_sampler * lhm_sampler_init_grammar_lazy_patterns(
        const struct lhm_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const lhm_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    struct lhm_sampler * lhm_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w
    struct lhm_sampler * lhm_sampler_init_dry(
            const struct lhm_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    struct lhm_sampler * lhm_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    struct lhm_sampler * lhm_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const lhm_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    struct lhm_sampler * lhm_sampler_init_infill(const struct lhm_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LHM_DEFAULT_SEED otherwise
    uint32_t lhm_sampler_get_seed(const struct lhm_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = lhm_get_logits_ith(ctx, idx);
    //    lhm_token_data_array cur_p = { ... init from logits ... };
    //    lhm_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    lhm_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    lhm_token lhm_sampler_sample(struct lhm_sampler * smpl, struct lhm_context * ctx, int32_t idx);

    // TODO: extend in the future
    //void lhm_decode_with_sampler(struct lhm_context * ctx, struct lhm_sampler * smpl, struct lhm_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          lhm_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    int32_t lhm_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          lhm_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    int32_t lhm_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

    // Print system information
    const char * lhm_print_system_info(void);

    //
    // Performance utils
    //
    // NOTE: Used by lhm.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct lhm_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
    };

    struct lhm_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    struct lhm_perf_context_data lhm_perf_context      (const struct lhm_context * ctx);
    void                           lhm_perf_context_print(const struct lhm_context * ctx);
    void                           lhm_perf_context_reset(      struct lhm_context * ctx);

    // NOTE: the following work only with samplers constructed via lhm_sampler_chain_init
    struct lhm_perf_sampler_data lhm_perf_sampler      (const struct lhm_sampler * chain);
    void                           lhm_perf_sampler_print(const struct lhm_sampler * chain);
    void                           lhm_perf_sampler_reset(      struct lhm_sampler * chain);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*lhm_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    bool lhm_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct lhm_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in lhm_context if 0

        lhm_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    void lhm_opt_init(struct lhm_context * lctx, struct lhm_model * model, struct lhm_opt_params lopt_params);

    void lhm_opt_epoch(
            struct lhm_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

