#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class lhm_io_write_i {
public:
    lhm_io_write_i() = default;
    virtual ~lhm_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class lhm_io_read_i {
public:
    lhm_io_read_i() = default;
    virtual ~lhm_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
