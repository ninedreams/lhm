#pragma once

#include "lhm_kv_cache.h"

#include <vector>

//
// lhm_kv_cache_dsa
//

// utilizes two instances of lhm_kv_cache:
// - the first instance is for caching key tensors of the model,
// - the second instance is for caching lightning indexer key tensors

class lhm_kv_cache_dsa : public lhm_memory_i {
public:
    lhm_kv_cache_dsa(
            const lhm_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               lhm_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~lhm_kv_cache_dsa() = default;

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
    void state_read (lhm_io_read_i  & io, lhm_seq_id seq_id = -1, lhm_state_seq_flags flags = 0) override;

    //
    // lhm_kv_cache_dsa specific API
    //

    lhm_kv_cache * get_mla() const;
    lhm_kv_cache * get_lid() const;

private:
    // we keep indexer KV cache hparams instance here as lhm_kv_cache stores only reference to it
    lhm_hparams hparams_lid;
    const uint32_t n_stream  = 1;

    std::unique_ptr<lhm_kv_cache> kv_mla;
    std::unique_ptr<lhm_kv_cache> kv_lid;
};

class lhm_kv_cache_dsa_context : public lhm_memory_context_i {
public:
    using slot_info_vec_t = lhm_kv_cache::slot_info_vec_t;

    // used for errors
    lhm_kv_cache_dsa_context(lhm_memory_status status);

    // used to create a full-cache context
    lhm_kv_cache_dsa_context(
            lhm_kv_cache_dsa * kv);

    // used to create an update context
    lhm_kv_cache_dsa_context(
            lhm_kv_cache_dsa * kv,
            lhm_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    lhm_kv_cache_dsa_context(
            lhm_kv_cache_dsa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_ik,
            std::vector<lhm_ubatch> ubatches);

    virtual ~lhm_kv_cache_dsa_context();

    //
    // lhm_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    lhm_memory_status  get_status() const override;
    const lhm_ubatch & get_ubatch() const override;

    //
    // lhm_kv_cache_dsa_context specific API
    //

    const lhm_kv_cache_context * get_mla() const;
    const lhm_kv_cache_context * get_lid()  const;

private:
    //lhm_kv_cache_dsa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<lhm_ubatch> ubatches;

    const lhm_memory_context_ptr ctx_mla;
    const lhm_memory_context_ptr ctx_lid;

    const lhm_memory_status status;
};
