#pragma once

#include "base_models.h"

struct lhm_model_deepseek4 : public lhm_model_base {
    lhm_model_deepseek4(const struct lhm_model_params & params) : lhm_model_base(params) {}
    void load_arch_hparams(lhm_model_loader & ml) override;
    void load_arch_tensors(lhm_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const lhm_model & model, const llm_graph_params & params);

        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                ggml_tensor ** post,
                ggml_tensor ** comb,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_head(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base) const;

        ggml_tensor * build_attention(
                const lhm_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_hca_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t n_embd_head,
                const char * name,
                int il) const;

        ggml_tensor * build_overlap_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t ratio,
                int64_t n_embd_head,
                const char * name,
                int il) const;

        ggml_tensor * build_lid_top_k(
                const lhm_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_top_k_mask(
                ggml_tensor * kq_mask,
                ggml_tensor * top_k,
                const char * name,
                int il) const;

        ggml_tensor * build_csa_lid_attention(
                const lhm_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_hca_attention(
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_raw_attention(
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_hc_weighted_sum(
                ggml_tensor * x,
                ggml_tensor * weights) const;

        ggml_tensor * build_hc_sinkhorn(
                ggml_tensor * comb,
                int il) const;
    };

    struct graph_mtp : public graph {
        graph_mtp(const lhm_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
