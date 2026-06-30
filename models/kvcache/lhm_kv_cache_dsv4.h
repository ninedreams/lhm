#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "lhm_kv_cache.h"
#include "lhm_kv_cache_iswa.h"

class lhm_dsv4_comp_state {
public:
    lhm_dsv4_comp_state(
            const lhm_model & model,
            bool            offload,
            bool            unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
            uint32_t        state_size,
            uint32_t        n_embd_state,
            const char    * name,
        const lhm_memory_i::layer_filter_cb & filter);

    void clear(bool data);

    uint32_t get_ratio()    const;
    uint32_t get_state_size() const;
    uint32_t get_n_stream() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    void state_write(lhm_io_write_i & io, lhm_seq_id seq_id, lhm_state_seq_flags flags) const;
    void state_read (lhm_io_read_i  & io, lhm_seq_id seq_id, lhm_state_seq_flags flags);

    ggml_tensor * get_kv   (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_score(ggml_context * ctx, int32_t il) const;

    ggml_tensor * cpy_kv   (ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;

private:
    struct layer {
        uint32_t il;

        ggml_tensor * kv;
        ggml_tensor * score;
    };

    const uint32_t ratio;
    const uint32_t state_size;
    const uint32_t n_embd_state;
    const uint32_t n_stream;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    std::vector<layer> layers;

    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;
};

//
// lhm_kv_cache_dsv4
//

// DSV4 uses a normal raw/SWA token cache plus compressed K-only block caches.
// The compressed caches are storage only; DSV4-specific visibility and block
// planning are handled by lhm_kv_cache_dsv4_context / llm_graph_input_dsv4.

class lhm_kv_cache_dsv4 : public lhm_memory_i {
public:
    lhm_kv_cache_dsv4(
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
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~lhm_kv_cache_dsv4() = default;

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

    void state_write(lhm_io_write_i & io, lhm_seq_id seq_id = -1, lhm_state_seq_flags flags = 0) const override;
    void state_read (lhm_io_read_i  & io, lhm_seq_id seq_id = -1, lhm_state_seq_flags flags = 0) override;

    //
    // lhm_kv_cache_dsv4 specific API
    //

    lhm_kv_cache_iswa * get_raw() const;
    lhm_kv_cache      * get_csa() const;
    lhm_kv_cache      * get_hca() const;
    lhm_kv_cache      * get_lid() const;
    lhm_dsv4_comp_state * get_csa_state() const;
    lhm_dsv4_comp_state * get_hca_state() const;
    lhm_dsv4_comp_state * get_lid_state() const;

private:
    lhm_hparams hparams_raw;
    lhm_hparams hparams_csa;
    lhm_hparams hparams_hca;
    lhm_hparams hparams_lid;

    const uint32_t n_seq_max;

    std::unique_ptr<lhm_kv_cache_iswa> kv_raw;
    std::unique_ptr<lhm_kv_cache>      kv_csa;
    std::unique_ptr<lhm_kv_cache>      kv_hca;
    std::unique_ptr<lhm_kv_cache>      kv_lid;
    std::unique_ptr<lhm_dsv4_comp_state> csa_state;
    std::unique_ptr<lhm_dsv4_comp_state> hca_state;
    std::unique_ptr<lhm_dsv4_comp_state> lid_state;

    void clear_compressed(bool data);
};

// DSV4 raw attention only uses the SWA half of kv_raw. The base half is kept
// for generic ISWA bookkeeping, but it has no DSV4 layers to expose here.
class lhm_kv_cache_dsv4_raw_context : public lhm_memory_context_i {
public:
    using slot_info_vec_t = lhm_kv_cache::slot_info_vec_t;

    lhm_kv_cache_dsv4_raw_context(lhm_kv_cache_iswa * kv);

    lhm_kv_cache_dsv4_raw_context(
            lhm_kv_cache_iswa * kv,
            lhm_context * lctx,
            bool optimize);

    lhm_kv_cache_dsv4_raw_context(
            lhm_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base_write,
            slot_info_vec_t sinfos_swa_write,
            slot_info_vec_t sinfos_swa_read,
            std::vector<lhm_ubatch> ubatches,
            std::vector<lhm_ubatch> ubatches_write);

    bool next() override;
    bool apply() override;

    lhm_memory_status get_status() const override;
    const lhm_ubatch & get_ubatch() const override;

    uint32_t get_n_kv() const;
    uint32_t get_n_write() const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const lhm_ubatch & ubatch) const;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst) const;
    void set_input_kq_mask(ggml_tensor * dst, const lhm_ubatch * ubatch, bool causal_attn) const;
    void set_input_k_rot(ggml_tensor * dst) const;

private:
    size_t i_next = 0;

    lhm_kv_cache * kv_swa = nullptr;

    slot_info_vec_t sinfos_write;
    slot_info_vec_t sinfos_read;
    std::vector<lhm_ubatch> ubatches;
    std::vector<lhm_ubatch> ubatches_write;

    const lhm_memory_context_ptr ctx_base_mem;
    const lhm_memory_context_ptr ctx_swa_mem;

    uint32_t n_kv = 0;

    const lhm_memory_status status;
};

// DSV4 compressed KV rows are graph outputs, not normal token KV writes.
// Keep a small context that exposes K tensors without generic apply() semantics.
class lhm_kv_cache_dsv4_comp_context {
public:
    using slot_info_vec_t = lhm_kv_cache::slot_info_vec_t;

    lhm_kv_cache_dsv4_comp_context(lhm_kv_cache * kv);

    lhm_kv_cache_dsv4_comp_context(
            lhm_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<lhm_ubatch> ubatches);

    bool next();

    uint32_t get_n_kv() const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    void set_input_k_rot(ggml_tensor * dst) const;

private:
    lhm_kv_cache * kv;

    size_t i_cur = 0;
    slot_info_vec_t sinfos;
    std::vector<lhm_ubatch> ubatches;

    uint32_t n_kv;
};

class lhm_kv_cache_dsv4_context : public lhm_memory_context_i {
public:
    using slot_info_vec_t = lhm_kv_cache::slot_info_vec_t;

    struct comp_plan {
        // Per-ubatch recipe for updating compressor state, committing completed
        // compressed rows, and masking the compressed attention source.

        // APE row ids, i.e. pos % ratio, for the compressor-state updates.
        std::vector<int32_t> state_pos;

        // Current-ubatch source row ids and unique persistent-state
        // destination row ids for deterministic ring-state updates.
        std::vector<int32_t> state_persist_src_idxs;
        std::vector<int32_t> state_persist_dst_idxs;

        // Flattened source row ids used for state-backed commits. Source rows
        // index the graph-local [persistent_state | current_ubatch_scratch]
        // tensor. For overlapped compression the first half is previous rows
        // and the second half is current rows; a final synthetic zero/-inf row
        // may be addressed for the first block's previous half.
        std::vector<int32_t> state_read_idxs;

        // Final compressed-cache row ids written by state-backed commits.
        // A non-boundary CSA/LID decode step can target a masked scratch row.
        std::vector<int64_t> state_write_idxs;

        // RoPE positions for state-backed commits.
        std::vector<int32_t> state_write_pos;

        // Number of completed compressed rows visible for each query token.
        std::vector<int32_t> n_visible;

        // Number of streams used by the attention graph for this ubatch.
        int64_t n_stream = 1;

        // Graph-width for compressed rows. This can be larger than n_visible
        // so masked padding rows do not force a new graph at every CSA block.
        int64_t n_kv = 0;
    };

    lhm_kv_cache_dsv4_context(lhm_memory_status status);

    lhm_kv_cache_dsv4_context(
            lhm_kv_cache_dsv4 * kv);

    lhm_kv_cache_dsv4_context(
            lhm_kv_cache_dsv4 * kv,
            lhm_context * lctx,
            bool optimize);

    lhm_kv_cache_dsv4_context(
            lhm_kv_cache_dsv4 * kv,
            slot_info_vec_t sinfos_raw_base_write,
            slot_info_vec_t sinfos_raw_swa_write,
            slot_info_vec_t sinfos_raw_swa_read,
            std::vector<lhm_ubatch> ubatches,
            std::vector<lhm_ubatch> ubatches_raw);

    virtual ~lhm_kv_cache_dsv4_context();

    //
    // lhm_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    lhm_memory_status  get_status() const override;
    const lhm_ubatch & get_ubatch() const override;

    //
    // lhm_kv_cache_dsv4_context specific API
    //

    const lhm_kv_cache_dsv4_raw_context * get_raw() const;
    const lhm_kv_cache_dsv4_comp_context * get_csa() const;
    const lhm_kv_cache_dsv4_comp_context * get_hca() const;
    const lhm_kv_cache_dsv4_comp_context * get_lid() const;
    const lhm_dsv4_comp_state       * get_csa_state() const;
    const lhm_dsv4_comp_state       * get_hca_state() const;
    const lhm_dsv4_comp_state       * get_lid_state() const;

    const comp_plan & get_csa_plan() const;
    const comp_plan & get_hca_plan() const;
    const comp_plan & get_lid_plan() const;

    const comp_plan & get_csa_plan(const lhm_ubatch & ubatch) const;
    const comp_plan & get_hca_plan(const lhm_ubatch & ubatch) const;
    const comp_plan & get_lid_plan(const lhm_ubatch & ubatch) const;

private:
    size_t i_next = 0;

    std::vector<lhm_ubatch> ubatches;

    std::vector<comp_plan> plans_csa;
    std::vector<comp_plan> plans_hca;
    std::vector<comp_plan> plans_lid;

    const std::unique_ptr<lhm_kv_cache_dsv4_raw_context> ctx_raw;
    const lhm_memory_context_ptr ctx_csa_mem;
    const lhm_memory_context_ptr ctx_hca_mem;
    const lhm_memory_context_ptr ctx_lid_mem;

    const std::unique_ptr<lhm_kv_cache_dsv4_comp_context> ctx_csa;
    const std::unique_ptr<lhm_kv_cache_dsv4_comp_context> ctx_hca;
    const std::unique_ptr<lhm_kv_cache_dsv4_comp_context> ctx_lid;

    const lhm_dsv4_comp_state * csa_state = nullptr;
    const lhm_dsv4_comp_state * hca_state = nullptr;
    const lhm_dsv4_comp_state * lid_state = nullptr;

    bool reserve_plans = false;
    mutable comp_plan reserve_plan_csa;
    mutable comp_plan reserve_plan_hca;
    mutable comp_plan reserve_plan_lid;

    const lhm_memory_status status;
};
