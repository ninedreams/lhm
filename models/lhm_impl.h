#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include <ggml.h>


//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

template <typename dst_t, typename src_t>
static inline dst_t lhm_cast(src_t v) {
    if constexpr (std::is_same_v<src_t, dst_t>) {
        return v;
    } else if constexpr (std::is_same_v<src_t, ggml_fp16_t> && std::is_same_v<dst_t, float>) {
        return ggml_fp16_to_fp32(v);
    } else if constexpr (std::is_same_v<src_t, float> && std::is_same_v<dst_t, ggml_fp16_t>) {
        return ggml_fp32_to_fp16(v);
    } else {
        static_assert(std::is_same_v<dst_t, void>, "unsupported type combination");
    }
}

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to lhm_format ?
std::string format(const char * fmt, ...);

std::string lhm_format_tensor_shape(const std::vector<int64_t> & ne);
std::string lhm_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define LHM_TENSOR_NAME_FATTN   "__fattn__"
#define LHM_TENSOR_NAME_FGDN_AR "__fgdn_ar__"
#define LHM_TENSOR_NAME_FGDN_CH "__fgdn_ch__"
