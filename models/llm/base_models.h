
#pragma once

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

#include "graph/lhm_graph.h"
#include "graph/lhm_graph_dsv4.h"
#include "loader/lhm_model_loader.h"

#include "lhm_model.h"

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const lhm_model & model, const lhm_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const lhm_model & model, const lhm_ubatch & ubatch, int il) const;

};

// base class for delta-net models (Qwen-35, Qwen-35-MoE)
// or other common classes that include here
struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
    // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
    ggml_tensor * build_conv_state(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        qkv_mixed,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};
