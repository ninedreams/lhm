#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>
#include <vector>

#include "lhm.h"

using lhm_tokens = std::vector<lhm_token>;

struct lhm_model_deleter {
    void operator()(lhm_model * model) { lhm_model_free(model); }
};

struct lhm_context_deleter {
    void operator()(lhm_context * context) { lhm_free(context); }
};

struct lhm_sampler_deleter {
    void operator()(lhm_sampler * sampler) { lhm_sampler_free(sampler); }
};

struct lhm_adapter_lora_deleter {
    void operator()(lhm_adapter_lora * adapter) { lhm_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<lhm_model, lhm_model_deleter> lhm_model_ptr;
typedef std::unique_ptr<lhm_context, lhm_context_deleter> lhm_context_ptr;
typedef std::unique_ptr<lhm_sampler, lhm_sampler_deleter> lhm_sampler_ptr;
typedef std::unique_ptr<lhm_adapter_lora, lhm_adapter_lora_deleter> lhm_adapter_lora_ptr;
