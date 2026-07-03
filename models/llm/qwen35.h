#pragma once

#include "base_models.h"

struct lhm_model_qwen35 : public lhm_model_base {
    lhm_model_qwen35(const struct lhm_model_params & params) : lhm_model_base(params) {}
    void load_arch_hparams(lhm_model_loader & ml) override;
    void load_arch_tensors(lhm_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const lhm_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const lhm_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const lhm_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
