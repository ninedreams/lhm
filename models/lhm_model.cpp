#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-cpp.h>

#include "loader/lhm_model_loader.h"
#include "params/lhm_hparams.h"
#include "kvcache/lhm_kv_cache.h"
#include "kvcache/lhm_kv_cache_iswa.h"
#include "kvcache/lhm_kv_cache_dsa.h"
#include "kvcache/lhm_kv_cache_dsv4.h"
#include "memory/lhm_memory_hybrid.h"
#include "memory/lhm_memory_hybrid_iswa.h"
#include "memory/lhm_memory_recurrent.h"
#include "llm/models.h"

#include "lhm_model.h"
#include "lhm_arch.h"
#include "lhm_ext.h"
#include "lhm_impl.h"
#include "lhm_mmap.h"
#include "lhm_cparams.h"

static lhm_model * lhm_model_mapping(llm_arch arch, const lhm_model_params & params) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
            return new lhm_model_qwen35(params);
        case LLM_ARCH_QWEN35MOE:
            return new lhm_model_qwen35moe(params);
        case LLM_ARCH_DEEPSEEK4:
            return new lhm_model_deepseek4(params);
        default:
            throw std::runtime_error(std::string("unsupported model architecture: '") + llm_arch_name(arch) + "'");
    }
}

lhm_model * lhm_model_create(llm_arch arch, const lhm_model_params & params) {
    lhm_model * model = lhm_model_mapping(arch, params);

    if (model != nullptr) {
        model->arch = arch;
        auto & devices = model->devices;
        if (!devices.empty() && devices[0].is_meta && !llm_arch_supports_sm_tensor(arch)) {
            throw std::runtime_error(std::string("LHM_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
        }
    }

    return model;
}

lhm_model * lhm_model_create(lhm_model_loader & ml, const lhm_model_params & params) {
    llm_arch arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }

    return lhm_model_create(arch, params);
}

struct ggml_backend_meta_split_state lhm_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    const lhm_meta_device_get_split_state_userdata * ud = (const lhm_meta_device_get_split_state_userdata *) userdata;
    const lhm_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;

    const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");

    const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    const std::regex pattern_r_cache         ("cache_r_l\\d*");
    const std::regex pattern_s_cache         ("cache_s_l\\d*");
    const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    const std::regex pattern_ffn_up_gate_weight("blk\\.\\d*\\.ffn_(up|gate)(_exps)?.weight");
    const std::regex pattern_ffn_up_gate_bias  ("blk\\.\\d*\\.ffn_(up|gate)(_exps)?.bias");
    const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    const std::regex pattern_ffn_down_bias     ("blk\\.\\d*\\.ffn_down.bias");
    const std::regex pattern_ffn_down_exps_bias("blk\\.\\d*\\.ffn_down_exps.bias");

    const std::regex pattern_output_weight("output\\.weight");
    const std::regex pattern_output_bias  ("output\\.bias");

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            LHM_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            LHM_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            LHM_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        LHM_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            LHM_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        if (ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    LHM_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_attn_gate_weight) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, head_ratio}};
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp;
                LHM_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {{n_ff_exp, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_embd;
            const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
            LHM_ASSERT(hparams.n_embd_k_gqa() == n_embd_gqa);
            LHM_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
            return {{n_embd, 1}, {n_embd_gqa, 2}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp;
            LHM_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {{n_ff_exp, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
        if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                LHM_ASSERT(segments.size() == 1);
                return {std::lcm(n_embd_q, blck_size_perf)/n_embd_q * n_gqa};
            }

            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                LHM_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                if (ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
                    return {std::lcm(2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                LHM_ASSERT(segments.size() == 1);
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                LHM_ASSERT(segments.size() == 1);
                return {granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                LHM_ASSERT(segments.size() == 2);
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_gate_weight) || std::regex_match(tensor_name, pattern_ffn_up_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) || std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            LHM_ASSERT(segments.size() == 1);
            return {blck_size_perf};
        }

        // everything else
        LHM_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t  ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t  g_s  = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
    }
    return split_state;
    GGML_UNUSED(userdata);
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_47M:           return "47M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_149M:          return "149M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_395M:          return "395M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_0_8B:          return "0.8B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_6B:          return "2.6B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_26B:           return "26B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_31B:           return "31B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_7B_A1B:        return "7B.A1B";
        case LLM_TYPE_8B_A1B:        return "8B.A1B";
        case LLM_TYPE_12B_A2_5B:     return "12B.A2.5B";
        case LLM_TYPE_16B_A1B:       return "16B.A1B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_24B_A2B:       return "24B.A2B";
        case LLM_TYPE_26B_A4B:       return "26B.A4B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LLM_TYPE_35B_A3B:       return "35B.A3B";
        case LLM_TYPE_48B_A3B:       return "48B.A3B";
        case LLM_TYPE_80B_A3B:       return "80B.A3B";
        case LLM_TYPE_100B_A6B:      return "100B.A6B";
        case LLM_TYPE_102B_A12B:     return "102B.A12B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_120B_A12B:     return "120B.A12B";
        case LLM_TYPE_122B_A10B:     return "122B.A10B";
        case LLM_TYPE_196B_A11B:     return "196B.A11B";
        case LLM_TYPE_230B_A10B:     return "230B.A10B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_310B_A15B:     return "310B.A15B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_397B_A17B:     return "397B.A17B";
        case LLM_TYPE_685B_A37B:     return "685B.A37B";
        case LLM_TYPE_744B_A40B:     return "744B.A40B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * lhm_expert_gating_func_name(lhm_expert_gating_func_type type) {
    switch (type) {
        case LHM_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LHM_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        case LHM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default:                                    return "unknown";
    }
}

static const std::map<lhm_rope_scaling_type, const char *> LHM_ROPE_SCALING_TYPES = {
    { LHM_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LHM_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LHM_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LHM_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string lhm_rope_scaling_type_name(lhm_rope_scaling_type rope_scaling_type) {
    return LHM_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static lhm_rope_scaling_type lhm_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LHM_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (lhm_rope_scaling_type) kv.first;
        }
    }

    return LHM_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// Maps the GGUF `<arch>.hidden_activation` string to the FFN op type used by the
// graph builders. Only gated activations that map cleanly to llm_ffn_op_type are
// listed; unrecognized values fall back to GeGLU, which matches the historical
// default for ModernBert-style architectures.
static const std::map<std::string, llm_ffn_op_type> LLM_FFN_OP_TYPES_FROM_STRING = {
    { "gelu",   LLM_FFN_GEGLU  },
    { "geglu",  LLM_FFN_GEGLU  },
    { "silu",   LLM_FFN_SWIGLU },
    { "swish",  LLM_FFN_SWIGLU },
    { "swiglu", LLM_FFN_SWIGLU },
    { "relu",   LLM_FFN_RELU   },
    { "reglu",  LLM_FFN_REGLU  },
};

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback) {
    const auto it = LLM_FFN_OP_TYPES_FROM_STRING.find(name);
    if (it != LLM_FFN_OP_TYPES_FROM_STRING.end()) {
        return it->second;
    }
    return fallback;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<lhm_device> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(fmt::format("{}: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LHM_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, lhm_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LHM_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(fmt::format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct lhm_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    lhm_mmaps mappings;

    // objects representing data potentially being locked in memory
    lhm_mlocks mlock_bufs;
    lhm_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;
};

lhm_model::lhm_model(const lhm_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

lhm_model::~lhm_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

void lhm_model_base::load_stats(lhm_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void lhm_model_base::load_hparams(lhm_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,       hparams.n_expert_used,   false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    LHM_ASSERT(hparams.n_expert <= LHM_MAX_EXPERTS);
    LHM_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        LHM_ASSERT(hparams.n_expert_used > 0);
        LHM_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            LHM_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            LHM_ASSERT(hparams.n_group_used > 0);
            LHM_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        LHM_ASSERT(hparams.n_expert_used == 0);
        LHM_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.is_swa_impl.begin(),   hparams.is_swa_impl.end(), 0);
    std::fill(hparams.is_recr_impl.begin(),  hparams.is_recr_impl.end(),  llm_arch_is_recurrent(ml.get_arch()) ? 1 : 0);

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(),    hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(),     hparams.xielu_eps.end(), 0.0f);

    std::fill(hparams.swiglu_clamp_exp.begin(),   hparams.swiglu_clamp_exp.end(),   0.0f);
    std::fill(hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.end(), 0.0f);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer(), false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer(), false);

    // Populate deepstack_mapping_arr - initialized to -1 (no deepstack)
    std::fill(hparams.deepstack_mapping_arr.begin(), hparams.deepstack_mapping_arr.end(), -1);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer(), false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = lhm_rope_scaling_type_from_string(rope_scaling);
    LHM_ASSERT(hparams.rope_scaling_type_train != LHM_ROPE_SCALING_TYPE_UNSPECIFIED);

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_ALPHA,       hparams.rope_scaling_alpha, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);
    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // per-arch hparams
    load_arch_hparams(ml);

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = lhm_model_rope_type(this);
}

void lhm_model_base::load_vocab(lhm_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool lhm_model_base::load_tensors(lhm_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer_all = hparams.n_layer_all;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    LOG_INFO("loading model tensors, this can take a while... (mmap = {}, direct_io = {})", ml.use_mmap ? "true" : "false", ml.use_direct_io ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(fmt::format("{}: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
    auto get_layer_buft_list = [&](int il) -> lhm_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LOG_DEBUG("load_tensors: layer {:3d} assigned to device {}, is_swa = {:d}", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LOG_DEBUG("load_tensors: layer {:3d} assigned to device {}, is_swa = {:d}", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);

    const auto TENSOR_NOT_REQUIRED = lhm_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        // TODO stupid, change to fmt
        const auto tn = LLM_TN(arch);

        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;

        if (n_expert > 0 && n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer_all);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer_all; ++i) {
            auto & layer = layers[i];

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // recurrent / linear-attention weight scales (per-tensor, shape {1})
            if (!layer.ssm_in_s && layer.ssm_in) {
                layer.ssm_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_s && layer.ssm_out) {
                layer.ssm_out_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_s && layer.ssm_alpha) {
                layer.ssm_alpha_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_s && layer.ssm_beta) {
                layer.ssm_beta_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_in_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_in_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_in_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_in_in_s && layer.ssm_in) {
                layer.ssm_in_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_in_s && layer.ssm_out) {
                layer.ssm_out_in_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_in_s && layer.ssm_alpha) {
                layer.ssm_alpha_in_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_in_s && layer.ssm_beta) {
                layer.ssm_beta_in_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_in_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_in_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }
        // output scales
        if (output && output->type == GGML_TYPE_NVFP4) {
            // weight scale
            if (!output_s) {
                output_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
            }
            // input scale
            if (!output_in_s) {
                output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
            }
        }
    }
    ml.done_getting_tensors();

    // Tied NVFP4 output is valid when no separate LM-head scale tensors are present.
    // If sidecar scales exist, the output weight must be an actual output tensor.
    LHM_ASSERT(!(output && tok_embd &&
            strcmp(output->name, tok_embd->name) == 0 &&
            output->type == GGML_TYPE_NVFP4 &&
            (output_s || output_in_s)));
    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, lhm_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ml.ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ml.ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [buft, ctx_ptr] : ml.ctx_map) {
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        lhm_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(fmt::format("{}: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            LHM_ASSERT(!ml.no_alloc);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(fmt::format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(fmt::format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new lhm_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_buf_maps.emplace_back(ctx, buf_map);
    }

    if (lhm_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_all);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LOG_INFO("offloading output layer to GPU");
            n_repeating--;
        }
        LOG_INFO("offloading {:d} repeating layers to GPU", n_repeating);

        const int max_backend_supported_layers = n_layer_all + 1;
        const int max_offloadable_layers       = n_layer_all + 1;

        LOG_INFO("offloaded {:d}/{:d} layers to GPU", std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LOG_INFO("{:12s} model buffer size = {:8.2f} MiB", ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

ggml_tensor * lhm_model_base::create_tensor(lhm_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}

std::string lhm_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string lhm_model::type_name() const {
    return llm_type_name(type);
}

std::string lhm_model::desc() const {
    return pimpl->desc_str;
}

size_t lhm_model::size() const {
    return pimpl->n_bytes;
}

size_t lhm_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t lhm_model::n_devices() const {
    return devices.size();
}

const float * lhm_model::tensor_split() const {
    return params.tensor_split;
}

uint32_t lhm_model::n_gpu_layers() const {
    // note: plus 1 for the "output" layer
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}

lhm_split_mode lhm_model::split_mode() const {
    return params.split_mode;
}

std::map<ggml_backend_buffer_type_t, size_t> lhm_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            LHM_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            LHM_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // LHM_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t lhm_model::n_elements() const {
    return pimpl->n_elements;
}

void lhm_model::print_info() const {
    const std::string rope_scaling_type = lhm_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<int32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<int32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LOG_INFO("arch                  = {}", arch_name().c_str());
    LOG_INFO("vocab_only            = {:d}", hparams.vocab_only);
    LOG_INFO("no_alloc              = {:d}", hparams.no_alloc);

    if (!hparams.vocab_only) {
        LOG_INFO("n_ctx_train           = {:d}", hparams.n_ctx_train);
        LOG_INFO("n_embd_inp            = {:d}", hparams.n_embd_inp());
        LOG_INFO("n_embd                = {:d}", hparams.n_embd);
        LOG_INFO("n_embd_out            = {:d}", hparams.n_embd_out());
        LOG_INFO("n_layer               = {:d}", hparams.n_layer());
        LOG_INFO("n_layer_all           = {:d}", hparams.n_layer_all);
        LOG_INFO("n_head                = {}", print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer_all).c_str());
        LOG_INFO("n_head_kv             = {}", print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer_all).c_str());
        LOG_INFO("n_rot                 = {:d}", hparams.n_rot_full);
        LOG_INFO("n_swa                 = {:d}", hparams.n_swa);
        LOG_INFO("is_swa_any            = {:d}", hparams.is_swa_any());
        LOG_INFO("n_embd_head_k         = {:d}", hparams.n_embd_head_k_full);
        LOG_INFO("n_embd_head_v         = {:d}", hparams.n_embd_head_v_full);
        LOG_INFO("n_gqa                 = {}", print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer_all).c_str());
        LOG_INFO("n_embd_k_gqa          = {}", print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer_all).c_str());
        LOG_INFO("n_embd_v_gqa          = {}", print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer_all).c_str());
        LOG_INFO("f_norm_eps            = {:.1e}", hparams.f_norm_eps);
        LOG_INFO("f_norm_rms_eps        = {:.1e}", hparams.f_norm_rms_eps);
        LOG_INFO("f_clamp_kqv           = {:.1e}", hparams.f_clamp_kqv);
        LOG_INFO("f_max_alibi_bias      = {:.1e}", hparams.f_max_alibi_bias);
        LOG_INFO("f_logit_scale         = {:.1e}", hparams.f_logit_scale);
        LOG_INFO("f_attn_scale          = {:.1e}", hparams.f_attention_scale);
        LOG_INFO("f_attn_value_scale    = {:.4f}", hparams.f_attn_value_scale);
        LOG_INFO("n_ff                  = {}", print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer_all).c_str());
        LOG_INFO("n_expert              = {:d}", hparams.n_expert);
        LOG_INFO("n_expert_used         = {:d}", hparams.n_expert_used);
        LOG_INFO("n_expert_groups       = {:d}", hparams.n_expert_groups);
        LOG_INFO("n_group_used          = {:d}", hparams.n_group_used);
        LOG_INFO("causal attn           = {:d}", hparams.causal_attn);
        LOG_INFO("pooling type          = {:d}", int(hparams.pooling_type));
        LOG_INFO("rope type             = {:d}", int(hparams.rope_type));
        LOG_INFO("rope scaling          = {}", rope_scaling_type.c_str());
        LOG_INFO("freq_base_train       = {:.1f}", hparams.rope_freq_base_train);
        LOG_INFO("freq_scale_train      = {}", hparams.rope_freq_scale_train);
        if (hparams.swa_type != LHM_SWA_TYPE_NONE) {
            LOG_INFO("freq_base_swa         = {:.1f}", hparams.rope_freq_base_train_swa);
            LOG_INFO("freq_scale_swa        = {}", hparams.rope_freq_scale_train_swa);
            LOG_INFO("n_embd_head_k_swa     = {:d}", hparams.n_embd_head_k_swa);
            LOG_INFO("n_embd_head_v_swa     = {:d}", hparams.n_embd_head_v_swa);
            LOG_INFO("n_rot_swa             = {:d}", hparams.n_rot_swa);
        }
        LOG_INFO("n_ctx_orig_yarn       = {:d}", hparams.n_ctx_orig_yarn);
        LOG_INFO("rope_yarn_log_mul     = {:.4f}", hparams.rope_yarn_log_mul);
        LOG_INFO("rope_finetuned        = {}", hparams.rope_finetuned ? "yes" : "unknown");

        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LOG_INFO("mrope sections        = [{:d}, {:d}, {:d}, {:d}]", s[0], s[1], s[2], s[3]);
        }
        if (!classifier_labels.empty()) {
            LOG_INFO("n_cls_out             = {:d}", hparams.n_cls_out);

            size_t i = 0;
            for (const auto & label : classifier_labels) {
                LOG_INFO("cls_label[{:2d}]         = {}", i++, label.c_str());
            }
        }

        if (arch == LLM_ARCH_QWEN35 ||
            arch == LLM_ARCH_QWEN35MOE) {
            LOG_INFO("ssm_d_conv            = {:d}", hparams.ssm_d_conv);
            LOG_INFO("ssm_d_inner           = {:d}", hparams.ssm_d_inner);
            LOG_INFO("ssm_d_state           = {:d}", hparams.ssm_d_state);
            LOG_INFO("ssm_dt_rank           = {:d}", hparams.ssm_dt_rank);
            LOG_INFO("ssm_n_group           = {:d}", hparams.ssm_n_group);
            LOG_INFO("ssm_dt_b_c_rms        = {:d}", hparams.ssm_dt_b_c_rms);
        }

        LOG_INFO("model type            = {}", type_name().c_str());
        if (pimpl->n_elements >= 1e12) {
            LOG_INFO("model params          = {:.2f} T", pimpl->n_elements*1e-12);
        } else if (pimpl->n_elements >= 1e9) {
            LOG_INFO("model params          = {:.2f} B", pimpl->n_elements*1e-9);
        } else if (pimpl->n_elements >= 1e6) {
            LOG_INFO("model params          = {:.2f} M", pimpl->n_elements*1e-6);
        } else {
            LOG_INFO("model params          = {:.2f} K", pimpl->n_elements*1e-3);
        }

        // general kv
        LOG_INFO("general.name          = {}", name.c_str());
    }

    vocab.print_info();
}

ggml_backend_dev_t lhm_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t lhm_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(fmt::format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(fmt::format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t lhm_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

bool lhm_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * lhm_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float lhm_model::get_rope_freq_base (const lhm_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float lhm_model::get_rope_freq_scale(const lhm_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * lhm_model::get_rope_factors(const lhm_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

lhm_memory_i * lhm_model::create_memory(const lhm_memory_params & params, const lhm_cparams & cparams) const {
    lhm_memory_i * res;

    switch (arch) {
        default:
            {
                // The MTP head is dense-attention only on hybrid Qwen3.5/3.6/deepseekv4, so use a plain
                // attention KV cache for the MTP context instead of the hybrid wrapper.
                const bool mtp_on_hybrid_qwen35 =
                    params.ctx_type == LHM_CONTEXT_TYPE_MTP &&
                    (arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE);

                if (llm_arch_is_recurrent(arch)) {
                    res = new lhm_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            cparams.n_rs_seq,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen35) {
                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    lhm_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    lhm_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    if (arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE) {
                        filter_attn = [&](uint32_t il) {
                            return il < hparams.n_layer() && !hparams.is_recr(il);
                        };
                        filter_recr = [&](uint32_t il) {
                            return il < hparams.n_layer() && hparams.is_recr(il);
                        };
                    }

                    if (hparams.swa_type != LHM_SWA_TYPE_NONE) {
                        // Use hybrid-iswa for hybrid models with SWA
                        res = new lhm_memory_hybrid_iswa(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_swa_full     */ params.swa_full,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_ubatch     */ cparams.n_ubatch,
                            /* attn_n_pad        */ 1,
                            /* recurrent_type_r  */ GGML_TYPE_F32,
                            /* recurrent_type_s  */ GGML_TYPE_F32,
                            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    } else {
                        res = new lhm_memory_hybrid(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    }
                } else {
                    lhm_kv_cache::layer_filter_cb filter = nullptr;
                    lhm_memory_i::layer_reuse_cb reuse = nullptr;
                    lhm_kv_cache::layer_share_cb share = nullptr;

                    if (mtp_on_hybrid_qwen35) {
                        filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                    }


                    if (arch == LLM_ARCH_DEEPSEEK4) {
                        GGML_ASSERT(hparams.swa_type != LHM_SWA_TYPE_NONE);

                        res = new lhm_kv_cache_dsv4(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                params.swa_full,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                cparams.n_ubatch,
                                1,
                                filter,
                                reuse);
                    } else if (hparams.swa_type != LHM_SWA_TYPE_NONE) {
                        LHM_ASSERT(hparams.is_swa_any());

                        {
                            res = new lhm_kv_cache_iswa(
                                    *this,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    1,
                                    nullptr,
                                    filter,
                                    reuse,
                                    share);
                        }
                    } else {
                        LHM_ASSERT(!hparams.is_swa_any());

                        res = new lhm_kv_cache(
                                *this,
                                hparams,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                filter,
                                nullptr,
                                nullptr);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * lhm_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}


//
// interface implementation
//

lhm_model_params lhm_model_default_params() {
    lhm_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LHM_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_direct_io               =*/ false,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };

    return result;
}

const lhm_vocab * lhm_model_get_vocab(const lhm_model * model) {
    return &model->vocab;
}

void lhm_model_free(lhm_model * model) {
    delete model;
}

int32_t lhm_model_n_ctx_train(const lhm_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t lhm_model_n_embd(const lhm_model * model) {
    return model->hparams.n_embd;
}

int32_t lhm_model_n_embd_inp(const lhm_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t lhm_model_n_embd_out(const lhm_model * model) {
    return model->hparams.n_embd_out();
}

int32_t lhm_model_n_layer(const lhm_model * model) {
    return model->hparams.n_layer();
}

int32_t lhm_model_n_head(const lhm_model * model) {
    return model->hparams.n_head();
}

int32_t lhm_model_n_head_kv(const lhm_model * model) {
    return model->hparams.n_head_kv();
}

int32_t lhm_model_n_swa(const lhm_model * model) {
    // dsv4 kv-cache has SWA but it cannot be used as a rollback because of
    // other compression ratios, so we return 0 here
    if (model->arch == LLM_ARCH_DEEPSEEK4) {
        return 0;
    }
    return model->hparams.n_swa;
}


uint32_t lhm_model_n_cls_out(const struct lhm_model * model) {
    return model->hparams.n_cls_out;
}

const char * lhm_model_cls_label(const struct lhm_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t lhm_n_ctx_train(const lhm_model * model) {
    return lhm_model_n_ctx_train(model);
}

// deprecated
int32_t lhm_n_embd(const lhm_model * model) {
    return lhm_model_n_embd(model);
}

// deprecated
int32_t lhm_n_layer(const lhm_model * model) {
    return lhm_model_n_layer(model);
}

// deprecated
int32_t lhm_n_head(const lhm_model * model) {
    return lhm_model_n_head(model);
}

lhm_rope_type lhm_model_rope_type(const lhm_model * model) {
    switch (model->arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return LHM_ROPE_TYPE_IMROPE;
        case LLM_ARCH_DEEPSEEK4:
            return LHM_ROPE_TYPE_NORM;
        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            LHM_ABORT("unknown architecture");
    }

    return LHM_ROPE_TYPE_NONE;
}

float lhm_model_rope_freq_scale_train(const lhm_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t lhm_model_meta_val_str(const lhm_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t lhm_model_meta_count(const lhm_model * model) {
    return (int)model->gguf_kv.size();
}

const char * lhm_model_meta_key_str(lhm_model_meta_key key) {
    switch (key) {
        case LHM_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LHM_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LHM_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LHM_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LHM_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LHM_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LHM_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LHM_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LHM_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LHM_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LHM_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LHM_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t lhm_model_meta_key_by_index(const lhm_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t lhm_model_meta_val_str_by_index(const lhm_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t lhm_model_desc(const lhm_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

uint64_t lhm_model_size(const lhm_model * model) {
    return model->size();
}

const char * lhm_model_chat_template(const lhm_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end())
        return nullptr;

    return it->second.c_str();
}

uint64_t lhm_model_n_params(const lhm_model * model) {
    return model->n_elements();
}

// llm just have decoder
bool lhm_model_has_encoder(const lhm_model * model) {
    switch (model->arch) {
        default:
            return false;
    }
}

bool lhm_model_has_decoder(const lhm_model * model) {
    switch (model->arch) {
        default:
            return true;
    }
}

lhm_token lhm_model_decoder_start_token(const lhm_model * model) {
    return model->hparams.dec_start_token_id;
}

bool lhm_model_is_recurrent(const lhm_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool lhm_model_is_hybrid(const lhm_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool lhm_model_is_diffusion(const lhm_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & lhm_internal_get_tensor_map(const lhm_model * model) {
    return model->tensors_by_name;
}

int32_t lhm_model_n_expert(const struct lhm_model * model) {
    return model->hparams.n_expert;
}

int32_t lhm_model_n_devices(const struct lhm_model * model) {
    return (int32_t)model->devices.size();
}

ggml_backend_dev_t lhm_model_get_device(const struct lhm_model * model, int i) {
    if (i < 0 || i >= (int)model->devices.size()) {
        return nullptr;
    }
    return model->devices[i].dev;
}

//
// lhm_model_base
//

lhm_model_base::lhm_model_base(const struct lhm_model_params & params) : lhm_model(params), model(this), tn(model->arch),
    TENSOR_DUPLICATED     (lhm_model_loader::TENSOR_DUPLICATED),
    TENSOR_NOT_REQUIRED   (lhm_model_loader::TENSOR_NOT_REQUIRED),
    TENSOR_SKIP           (lhm_model_loader::TENSOR_SKIP),
    TENSOR_SKIP_IF_VIRTUAL(lhm_model_loader::TENSOR_SKIP_IF_VIRTUAL) {}

ggml_tensor * lhm_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    LHM_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}

void lhm_model_base::create_tensor_gate_up_exps(lhm_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
    layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED);
    if (layer.ffn_gate_up_exps == nullptr) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    }
}

void lhm_model_base::create_tensor_qkv(lhm_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;
    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    if (layer.wqkv) {
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    } else {
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
    }
}

const int32_t * lhm_model_target_layer_ids(const struct lhm_model * model) {
    const auto & v = model->target_layer_ids;
    return v.empty() ? nullptr : v.data();
}

uint32_t lhm_model_target_layer_ids_n(const struct lhm_model * model) {
    return (uint32_t) model->target_layer_ids.size();
}
