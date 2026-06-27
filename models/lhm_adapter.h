#pragma once

#include "lhm.h"

#include <ggml-cpp.h>

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// lhm_adapter_cvec
//

struct lhm_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const lhm_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const lhm_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using lhm_adapter_cvec_ptr = std::shared_ptr<lhm_adapter_cvec>;

//
// lhm_adapter_lora
//

struct lhm_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    lhm_adapter_lora_weight() = default;
    lhm_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct lhm_adapter_lora {
    lhm_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, lhm_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<lhm_token> alora_invocation_tokens;

    explicit lhm_adapter_lora(lhm_model * model) : model(model) {}
    ~lhm_adapter_lora() = default;

    lhm_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using lhm_adapter_loras = std::unordered_map<lhm_adapter_lora *, float>;
using lhm_adapter_loras_ptr = std::unique_ptr<lhm_adapter_loras>;
