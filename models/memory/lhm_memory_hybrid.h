#pragma once

#include <memory>
#include <vector>

#include "graph/lhm_graph.h"
#include "kvcache/lhm_kv_cache.h"

#include "lhm_batch.h"
#include "lhm_memory.h"
#include "lhm_memory_recurrent.h"

//
// lhm_memory_hybrid
//

// utilizes instances of lhm_memory_recurrent and lhm_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class lhm_memory_hybrid : public lhm_memory_i {
public:
    lhm_memory_hybrid(
        const lhm_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           lhm_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~lhm_memory_hybrid() = default;

    //
    // lhm_memory_i
    //

    lhm_memory_context_ptr init_batch(
            lhm_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    lhm_memory_context_ptr init_full() override;

    lhm_memory_context_ptr init_update(lhm_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (lhm_seq_id seq_id,                              lhm_pos p0, lhm_pos p1) override;
    void seq_cp  (lhm_seq_id seq_id_src, lhm_seq_id seq_id_dst, lhm_pos p0, lhm_pos p1) override;
    void seq_keep(lhm_seq_id seq_id)                                                          override;
    void seq_add (lhm_seq_id seq_id,                              lhm_pos p0, lhm_pos p1, lhm_pos shift) override;
    void seq_div (lhm_seq_id seq_id,                              lhm_pos p0, lhm_pos p1, int d) override;

    lhm_pos seq_pos_min(lhm_seq_id seq_id) const override;
    lhm_pos seq_pos_max(lhm_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(lhm_io_write_i & io, lhm_seq_id seq_id = -1, lhm_state_seq_flags flags = 0) const override;
    void state_read (lhm_io_read_i  & io, lhm_seq_id seq_id = -1, lhm_state_seq_flags flags = 0)       override;

    //
    // lhm_memory_hybrid specific API
    //

    lhm_kv_cache * get_mem_attn() const;
    lhm_memory_recurrent * get_mem_recr() const;

private:
    const lhm_hparams & hparams;

    const std::unique_ptr<lhm_kv_cache> mem_attn;
    const std::unique_ptr<lhm_memory_recurrent> mem_recr;
};

class lhm_memory_hybrid_context : public lhm_memory_context_i {
public:
    using slot_info_vec_t = lhm_kv_cache::slot_info_vec_t;

    // init failure
    explicit lhm_memory_hybrid_context(lhm_memory_status status);

    // init full
    explicit lhm_memory_hybrid_context(lhm_memory_hybrid * mem);

    // init update
    explicit lhm_memory_hybrid_context(
        lhm_memory_hybrid * mem,
              lhm_context * lctx,
                       bool   optimize);

    // init success
    lhm_memory_hybrid_context(
              lhm_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<lhm_ubatch>   ubatches);

    ~lhm_memory_hybrid_context() = default;

    bool next()  override;
    bool apply() override;

    lhm_memory_status  get_status() const override;
    const lhm_ubatch & get_ubatch() const override;

    //
    // lhm_memory_hybrid_context
    //

    const lhm_kv_cache_context * get_attn() const;
    const lhm_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<lhm_ubatch> ubatches;

    const lhm_memory_context_ptr ctx_attn;
    const lhm_memory_context_ptr ctx_recr;

    const lhm_memory_status status;
};
