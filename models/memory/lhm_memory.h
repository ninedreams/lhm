#pragma once

#include "lhm.h"
#include "lhm_graph.h"

#include <map>
#include <memory>
#include <functional>

struct lhm_ubatch;

class lhm_batch_allocr;

class lhm_io_write_i;
class lhm_io_read_i;

struct lhm_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    lhm_context_type ctx_type;

    lhm_memory_t mem_other;
};

enum lhm_memory_status {
    LHM_MEMORY_STATUS_SUCCESS = 0,
    LHM_MEMORY_STATUS_NO_UPDATE,
    LHM_MEMORY_STATUS_FAILED_PREPARE,
    LHM_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
lhm_memory_status lhm_memory_status_combine(lhm_memory_status s0, lhm_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool lhm_memory_status_is_fail(lhm_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - lhm_kv_cache_context
//   - lhm_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is lhm_memory_i::apply()
struct lhm_memory_context_i {
    virtual ~lhm_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const lhm_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual lhm_memory_status get_status() const = 0;
};

using lhm_memory_context_ptr = std::unique_ptr<lhm_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct lhm_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~lhm_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the lhm_memory_context_i::get_status() for the result
    virtual lhm_memory_context_ptr init_batch(
            lhm_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual lhm_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LHM_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual lhm_memory_context_ptr init_update(lhm_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (lhm_seq_id seq_id,                              lhm_pos p0, lhm_pos p1) = 0;
    virtual void seq_cp  (lhm_seq_id seq_id_src, lhm_seq_id seq_id_dst, lhm_pos p0, lhm_pos p1) = 0;
    virtual void seq_keep(lhm_seq_id seq_id) = 0;
    virtual void seq_add (lhm_seq_id seq_id,                              lhm_pos p0, lhm_pos p1, lhm_pos shift) = 0;
    virtual void seq_div (lhm_seq_id seq_id,                              lhm_pos p0, lhm_pos p1, int d) = 0;

    virtual lhm_pos seq_pos_min(lhm_seq_id seq_id) const = 0;
    virtual lhm_pos seq_pos_max(lhm_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(lhm_io_write_i & io, lhm_seq_id seq_id = -1, lhm_state_seq_flags flags = 0) const = 0;
    virtual void state_read (lhm_io_read_i  & io, lhm_seq_id seq_id = -1, lhm_state_seq_flags flags = 0) = 0;
};

using lhm_memory_ptr = std::unique_ptr<lhm_memory_i>;
