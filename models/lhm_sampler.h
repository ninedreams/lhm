#pragma once

#include "lhm.h"

#include <vector>

struct lhm_vocab;
struct lhm_grammar;

// sampler chain

struct lhm_sampler_chain {
    lhm_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        lhm_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for lhm_sampler_sample to avoid repeated allocations
    std::vector<lhm_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct lhm_sampler * lhm_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<lhm_token>> & seq_breakers);
