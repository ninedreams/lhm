#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct lhm_file;
struct lhm_mmap;
struct lhm_mlock;

using lhm_files  = std::vector<std::unique_ptr<lhm_file>>;
using lhm_mmaps  = std::vector<std::unique_ptr<lhm_mmap>>;
using lhm_mlocks = std::vector<std::unique_ptr<lhm_mlock>>;

// TODO maybe we do not need mmap?

struct lhm_file {
    lhm_file(const char * fname, const char * mode, bool use_direct_io = false);
    lhm_file(FILE * file);
    ~lhm_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct lhm_mmap {
    lhm_mmap(const lhm_mmap &) = delete;
    lhm_mmap(struct lhm_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~lhm_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct lhm_mlock {
    lhm_mlock();
    ~lhm_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t lhm_path_max();
