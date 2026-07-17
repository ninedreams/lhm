#include "lhm_kv_cache_dsa.h"

#include "lhm_impl.h"
#include "lhm_batch.h"
#include "lhm_model.h"

#include <algorithm>
#include <cassert>

//
// lhm_kv_cache_dsa
//

lhm_kv_cache_dsa::lhm_kv_cache_dsa(
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
    const  layer_reuse_cb & reuse) :
    hparams_lid(model.hparams), n_stream(unified ? 1 : n_seq_max) {

    LOG_INFO("creating main KV cache, size = {:d} cells", kv_size);

    kv_mla = std::make_unique<lhm_kv_cache>(
            model, model.hparams, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, nullptr, filter, reuse, nullptr);

    // we use lhm_kv_cache for caching indexer keys
    // by hand-tweaking some hparams we fool it to create
    // indexer key cache tensors with correct dimensions

    // DSA lightning indexer uses MQA with single key head
    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LHM_ROPE_TYPE_NEOX;

    LOG_INFO("creating indexer KV cache, size = {:d} cells", kv_size);

    kv_lid = std::make_unique<lhm_kv_cache>(
            model, hparams_lid, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, nullptr, filter, reuse, nullptr);
}

void lhm_kv_cache_dsa::clear(bool data) {
    kv_mla->clear(data);
    kv_lid->clear(data);
}

bool lhm_kv_cache_dsa::seq_rm(lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1) {
    bool res = true;

    res = res & kv_mla->seq_rm(seq_id, p0, p1);
    res = res & kv_lid->seq_rm(seq_id, p0, p1);

    return res;
}

void lhm_kv_cache_dsa::seq_cp(lhm_seq_id seq_id_src, lhm_seq_id seq_id_dst, lhm_pos p0, lhm_pos p1) {
    kv_mla->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_lid->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void lhm_kv_cache_dsa::seq_keep(lhm_seq_id seq_id) {
    kv_mla->seq_keep(seq_id);
    kv_lid->seq_keep(seq_id);
}

void lhm_kv_cache_dsa::seq_add(lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1, lhm_pos shift) {
    kv_mla->seq_add(seq_id, p0, p1, shift);
    kv_lid->seq_add(seq_id, p0, p1, shift);
}

void lhm_kv_cache_dsa::seq_div(lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1, int d) {
    kv_mla->seq_div(seq_id, p0, p1, d);
    kv_lid->seq_div(seq_id, p0, p1, d);
}

lhm_pos lhm_kv_cache_dsa::seq_pos_min(lhm_seq_id seq_id) const {
    return kv_mla->seq_pos_min(seq_id);
}

lhm_pos lhm_kv_cache_dsa::seq_pos_max(lhm_seq_id seq_id) const {
    return kv_mla->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> lhm_kv_cache_dsa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_mla->memory_breakdown();
    for (const auto & buft_size : kv_lid->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

lhm_memory_context_ptr lhm_kv_cache_dsa::init_batch(
            lhm_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<lhm_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_mla = kv_mla->prepare(ubatches);
        if (sinfos_mla.empty()) {
            break;
        }

        auto sinfos_lid = kv_lid->prepare(ubatches);
        if (sinfos_lid.empty()) {
            break;
        }

        assert(sinfos_mla.size() == sinfos_lid.size());

        return std::make_unique<lhm_kv_cache_dsa_context>(
                this, std::move(sinfos_mla), std::move(sinfos_lid), std::move(ubatches));
    } while (false);

    return std::make_unique<lhm_kv_cache_dsa_context>(LHM_MEMORY_STATUS_FAILED_PREPARE);
}

lhm_memory_context_ptr lhm_kv_cache_dsa::init_full() {
    return std::make_unique<lhm_kv_cache_dsa_context>(this);
}

lhm_memory_context_ptr lhm_kv_cache_dsa::init_update(lhm_context * lctx, bool optimize) {
    return std::make_unique<lhm_kv_cache_dsa_context>(this, lctx, optimize);
}

bool lhm_kv_cache_dsa::get_can_shift() const {
    return kv_mla->get_can_shift() &&
           kv_lid->get_can_shift() &&
           kv_mla->get_size() == kv_lid->get_size();
}

void lhm_kv_cache_dsa::state_write(lhm_io_write_i & io, lhm_seq_id seq_id, lhm_state_seq_flags flags) const {
    kv_mla->state_write(io, seq_id, flags);
    kv_lid->state_write(io, seq_id, flags);
}

void lhm_kv_cache_dsa::state_read(lhm_io_read_i & io, lhm_seq_id seq_id, lhm_state_seq_flags flags) {
    kv_mla->state_read(io, seq_id, flags);
    kv_lid->state_read(io, seq_id, flags);
}

lhm_kv_cache * lhm_kv_cache_dsa::get_mla() const {
    return kv_mla.get();
}

lhm_kv_cache * lhm_kv_cache_dsa::get_lid() const {
    return kv_lid.get();
}

//
// lhm_kv_cache_dsa_context
//

lhm_kv_cache_dsa_context::lhm_kv_cache_dsa_context(lhm_memory_status status) : status(status) {}

lhm_kv_cache_dsa_context::lhm_kv_cache_dsa_context(
        lhm_kv_cache_dsa * kv) :
    ctx_mla(kv->get_mla()->init_full()),
    ctx_lid(kv->get_lid()->init_full()),
    status(lhm_memory_status_combine(ctx_mla->get_status(), ctx_lid->get_status())) {
}

lhm_kv_cache_dsa_context::lhm_kv_cache_dsa_context(
        lhm_kv_cache_dsa * kv,
        lhm_context * lctx,
        bool optimize) :
    ctx_mla(kv->get_mla()->init_update(lctx, optimize)),
    ctx_lid(kv->get_lid()->init_update(lctx, optimize)),
    status(lhm_memory_status_combine(ctx_mla->get_status(), ctx_lid->get_status())) {
}

lhm_kv_cache_dsa_context::lhm_kv_cache_dsa_context(
        lhm_kv_cache_dsa * kv,
        slot_info_vec_t sinfos_mla,
        slot_info_vec_t sinfos_lid,
        std::vector<lhm_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_mla(new lhm_kv_cache_context(kv->get_mla(), std::move(sinfos_mla), this->ubatches)),
    ctx_lid(new lhm_kv_cache_context(kv->get_lid(), std::move(sinfos_lid), this->ubatches)),
    status(lhm_memory_status_combine(ctx_mla->get_status(), ctx_lid->get_status())) {
}

lhm_kv_cache_dsa_context:: ~lhm_kv_cache_dsa_context() = default;

bool lhm_kv_cache_dsa_context::next() {
    assert(status == LHM_MEMORY_STATUS_SUCCESS);

    ctx_mla->next();
    ctx_lid->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool lhm_kv_cache_dsa_context::apply() {
    assert(!lhm_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_mla->apply();
    res = res & ctx_lid->apply();

    return res;
}

lhm_memory_status lhm_kv_cache_dsa_context::get_status() const {
    return status;
}

const lhm_ubatch & lhm_kv_cache_dsa_context::get_ubatch() const {
    assert(status == LHM_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const lhm_kv_cache_context * lhm_kv_cache_dsa_context::get_mla() const {
    assert(status == LHM_MEMORY_STATUS_SUCCESS);

    return static_cast<const lhm_kv_cache_context *>(ctx_mla.get());
}

const lhm_kv_cache_context * lhm_kv_cache_dsa_context::get_lid()  const {
    assert(status == LHM_MEMORY_STATUS_SUCCESS);

    return static_cast<const lhm_kv_cache_context *>(ctx_lid.get());
}
