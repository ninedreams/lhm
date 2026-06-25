#pragma once

// this is a staging header for new lhm.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "lhm.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to lhm_graph_reserve.
 struct ggml_cgraph * lhm_graph_reserve(
        struct lhm_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
 ggml_type lhm_ftype_get_default_type(lhm_ftype ftype);

struct quantize_state_impl;

 quantize_state_impl * lhm_quant_init(
        const lhm_model * model,
        const lhm_model_quantize_params * params);

 void lhm_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct lhm_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with lhm_model_free().
 lhm_model * lhm_quant_model_from_metadata(const lhm_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
 bool lhm_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use lhm_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
 void lhm_quant_compute_types(
        quantize_state_impl * qs,
        lhm_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct lhm_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct lhm_device_memory_data {
    int64_t total;
    int64_t free;
    lhm_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using lhm_memory_breakdown = std::map<ggml_backend_buffer_type_t, lhm_memory_breakdown_data>;

 int32_t lhm_model_n_expert (const struct lhm_model * model);
 int32_t lhm_model_n_devices(const struct lhm_model * model);

 ggml_backend_dev_t lhm_model_get_device(const struct lhm_model * model, int i);

 lhm_memory_breakdown lhm_get_memory_breakdown(const struct lhm_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
 void lhm_set_embeddings_nextn(struct lhm_context * ctx, bool value, bool masked);

// mirrors:
//  float * lhm_get_embeddings(struct lhm_context * ctx);
 float * lhm_get_embeddings_nextn(struct lhm_context * ctx);

//  float * lhm_get_embeddings_ith(struct lhm_context * ctx, int32_t i);
 float * lhm_get_embeddings_nextn_ith(struct lhm_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
 void lhm_set_embeddings_layer_inp(struct lhm_context * ctx, uint32_t lid, bool value);

// mirrors:
//  float * lhm_get_embeddings(struct lhm_context * ctx);
 float * lhm_get_embeddings_layer_inp(struct lhm_context * ctx, uint32_t lid);

 lhm_context * lhm_get_ctx_other(struct lhm_context * ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
 const int32_t * lhm_model_target_layer_ids  (const struct lhm_model * model);
// returns the number of extracted layers from target model
 uint32_t        lhm_model_target_layer_ids_n(const struct lhm_model * model);
