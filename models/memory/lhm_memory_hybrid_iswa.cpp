#include "lhm_memory_hybrid_iswa.h"

#include "lhm_impl.h"
#include "lhm_model.h"
#include "lhm_context.h"

//
// lhm_memory_hybrid_iswa
//

lhm_memory_hybrid_iswa::lhm_memory_hybrid_iswa(
        const lhm_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
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
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new lhm_kv_cache_iswa(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        n_pad,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new lhm_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

lhm_memory_context_ptr lhm_memory_hybrid_iswa::init_batch(lhm_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<lhm_ubatch> ubatches;

        while (true) {
            lhm_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                if (mem_recr->n_rs_seq > 0) {
                    // [TAG_RECURRENT_ROLLBACK_SPLITS]
                    // TODO: recurrent state rollback does not support equal splits
                    ubatch = balloc.split_seq(n_ubatch);
                } else {
                    // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                    const bool unified = (mem_attn->get_base()->get_n_stream() == 1);
                    ubatch = balloc.split_equal(n_ubatch, !unified);
                }
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LOG_ERROR("failed to prepare recurrent ubatches");
            return std::make_unique<lhm_memory_hybrid_iswa_context>(LHM_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache (iswa version returns both base and swa slot infos)
        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            LOG_ERROR("failed to prepare attention base ubatches");
            return std::make_unique<lhm_memory_hybrid_iswa_context>(LHM_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_swa = mem_attn->get_swa()->prepare(ubatches);
        if (sinfos_swa.empty()) {
            LOG_ERROR("failed to prepare attention swa ubatches");
            return std::make_unique<lhm_memory_hybrid_iswa_context>(LHM_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<lhm_memory_hybrid_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while(false);

    return std::make_unique<lhm_memory_hybrid_iswa_context>(LHM_MEMORY_STATUS_FAILED_PREPARE);
}

lhm_memory_context_ptr lhm_memory_hybrid_iswa::init_full() {
    return std::make_unique<lhm_memory_hybrid_iswa_context>(this);
}

lhm_memory_context_ptr lhm_memory_hybrid_iswa::init_update(lhm_context * lctx, bool optimize) {
    return std::make_unique<lhm_memory_hybrid_iswa_context>(this, lctx, optimize);
}

bool lhm_memory_hybrid_iswa::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void lhm_memory_hybrid_iswa::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool lhm_memory_hybrid_iswa::seq_rm(lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void lhm_memory_hybrid_iswa::seq_cp(lhm_seq_id seq_id_src, lhm_seq_id seq_id_dst, lhm_pos p0, lhm_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void lhm_memory_hybrid_iswa::seq_keep(lhm_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void lhm_memory_hybrid_iswa::seq_add(lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1, lhm_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void lhm_memory_hybrid_iswa::seq_div(lhm_seq_id seq_id, lhm_pos p0, lhm_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

lhm_pos lhm_memory_hybrid_iswa::seq_pos_min(lhm_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

lhm_pos lhm_memory_hybrid_iswa::seq_pos_max(lhm_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> lhm_memory_hybrid_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void lhm_memory_hybrid_iswa::state_write(lhm_io_write_i & io, lhm_seq_id seq_id, lhm_state_seq_flags flags) const {
    mem_attn->state_write(io, seq_id, flags);
    mem_recr->state_write(io, seq_id, flags);
}

void lhm_memory_hybrid_iswa::state_read(lhm_io_read_i & io, lhm_seq_id seq_id, lhm_state_seq_flags flags) {
    mem_attn->state_read(io, seq_id, flags);
    mem_recr->state_read(io, seq_id, flags);
}

lhm_kv_cache_iswa * lhm_memory_hybrid_iswa::get_mem_attn() const {
    return mem_attn.get();
}

lhm_memory_recurrent * lhm_memory_hybrid_iswa::get_mem_recr() const {
    return mem_recr.get();
}

//
// lhm_memory_hybrid_iswa_context
//

lhm_memory_hybrid_iswa_context::lhm_memory_hybrid_iswa_context(lhm_memory_status status) : status(status) {}

lhm_memory_hybrid_iswa_context::lhm_memory_hybrid_iswa_context(lhm_memory_hybrid_iswa * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(lhm_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

lhm_memory_hybrid_iswa_context::lhm_memory_hybrid_iswa_context(
        lhm_memory_hybrid_iswa * mem,
                   lhm_context * lctx,
                            bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(lhm_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

lhm_memory_hybrid_iswa_context::lhm_memory_hybrid_iswa_context(
           lhm_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<lhm_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new lhm_kv_cache_iswa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_swa), this->ubatches)),
    ctx_recr(new lhm_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(lhm_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool lhm_memory_hybrid_iswa_context::next() {
    assert(status == LHM_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool lhm_memory_hybrid_iswa_context::apply() {
    assert(!lhm_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

lhm_memory_status lhm_memory_hybrid_iswa_context::get_status() const {
    return status;
}

const lhm_ubatch & lhm_memory_hybrid_iswa_context::get_ubatch() const {
    assert(status == LHM_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const lhm_kv_cache_iswa_context * lhm_memory_hybrid_iswa_context::get_attn() const {
    return static_cast<const lhm_kv_cache_iswa_context *>(ctx_attn.get());
}

const lhm_memory_recurrent_context * lhm_memory_hybrid_iswa_context::get_recr() const {
    return static_cast<const lhm_memory_recurrent_context *>(ctx_recr.get());
}
