#pragma once

#include "lhm_kv_cache.h"

#include <vector>

//
// lhm_kv_cache_iswa
//

// utilizes two instances of lhm_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class lhm_kv_cache_iswa : public lhm_memory_i {
public:
    lhm_kv_cache_iswa(
            const lhm_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
               lhm_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share);

    ~lhm_kv_cache_iswa() = default;

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
    // lhm_kv_cache_iswa specific API
    //

    lhm_kv_cache * get_base() const;
    lhm_kv_cache * get_swa () const;

private:
    const lhm_hparams & hparams;

    const bool unified;

    std::unique_ptr<lhm_kv_cache> kv_base;
    std::unique_ptr<lhm_kv_cache> kv_swa;
};

class lhm_kv_cache_iswa_context : public lhm_memory_context_i {
public:
    using slot_info_vec_t = lhm_kv_cache::slot_info_vec_t;

    // used for errors
    lhm_kv_cache_iswa_context(lhm_memory_status status);

    // used to create a full-cache context
    lhm_kv_cache_iswa_context(
            lhm_kv_cache_iswa * kv);

    // used to create an update context
    lhm_kv_cache_iswa_context(
            lhm_kv_cache_iswa * kv,
            lhm_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    lhm_kv_cache_iswa_context(
            lhm_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<lhm_ubatch> ubatches);

    virtual ~lhm_kv_cache_iswa_context();

    //
    // lhm_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    lhm_memory_status  get_status() const override;
    const lhm_ubatch & get_ubatch() const override;

    //
    // lhm_kv_cache_iswa_context specific API
    //

    const lhm_kv_cache_context * get_base() const;
    const lhm_kv_cache_context * get_swa()  const;

private:
    //lhm_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<lhm_ubatch> ubatches;

    const lhm_memory_context_ptr ctx_base;
    const lhm_memory_context_ptr ctx_swa;

    const lhm_memory_status status;
};
