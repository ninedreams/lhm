#include "lhm_graph_dsv4.h"


static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const lhm_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    float * data = (float *) dst->data;

    for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
        const int32_t n_visible = plan.n_visible[i];

        for (int64_t j = 0; j < dst->ne[0]; ++j) {
            data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
        }
    }
}

ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const lhm_kv_cache_dsv4_raw_context * mctx,
        const lhm_ubatch & ubatch,
        const lhm_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const bool use_fattn = cparams.flash_attn && (!cparams.kv_unified || n_stream == 1);
    const auto type = use_fattn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const lhm_kv_cache_dsv4_raw_context * mctx,
        const lhm_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const lhm_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    LOG_TRACE("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
            __func__, name, n_tokens, (int) n_stream,
            dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
            dsv4_plan_positions(plan.state_write_pos).c_str());
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const lhm_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const lhm_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const lhm_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const lhm_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const lhm_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", ubatch->n_tokens, n_stream);

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const lhm_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}
