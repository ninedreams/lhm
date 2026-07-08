#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include <allocator.h>
#include <client_service.h>
#include <types.h>

#include <ggml.h>

#include "graph/lhm_graph.h"
#include "memory/lhm_memory.h"

#include "lhm_context.h"
#include "lhm_model.h"
#include "lhm_batch.h"
#include "lhm_kv_cells.h"
#include "lhm_kv_cache.h"

struct lhm_cparams;
struct lhm_hparams;
struct lhm_model;
struct lhm_context;

// mooncake-store.md

// simple?
struct MooncakeSegmentInfo {
    void* base;
    size_t size;
};

class MooncakeClientWrapper;

struct ClientInfo {
    std::shared_ptr<MooncakeClientWrapper> client;
    std::unordered_map<std::string, void*> segments;
    std::string hostname;
};

/*
 * @brief A wrapper for the client.
 *
 * This class is used to wrap the client and provide a more convenient interface
 * for the mooncake client.
 */
class MooncakeClientWrapper {
   public:
    /**
     * @brief Constructor.
     *
     * @param client The client instance.
     * @param allocator Allocate slice memory for get and put operations.
     */
    MooncakeClientWrapper(std::shared_ptr<mooncake::Client> client,
                      std::shared_ptr<mooncake::SimpleAllocator> allocator);
    ~MooncakeClientWrapper();

    // The client wrapper is not copyable.
    MooncakeClientWrapper(const MooncakeClientWrapper&) = delete;
    MooncakeClientWrapper& operator=(const MooncakeClientWrapper&) = delete;

    /**
     * @brief Create a client wrapper.
     *
     * @param hostname The hostname of the client.
     * @param metadata_connstring Transfer engine metadata server url.
     * @param protocol Transfer protocol: rdma|tcp.
     * @param device_name The device name (used in transfer engine).
     * @param master_server_entry The master server entry.
     * @param local_buffer_size The local buffer size that will be used for get
     * and put operations.
     * @return The client wrapper.
     */
    static std::optional<std::shared_ptr<MooncakeClientWrapper>>
    CreateClientWrapper(const std::string& hostname,
                        const std::string& metadata_connstring,
                        const std::string& protocol,
                        const std::string& device_name,
                        const std::string& master_server_entry,
                        size_t local_buffer_size = 1024 * 1024 * 128);

    // Mount a segment. The buffer will be used to unmount the segment.
    mooncake::ErrorCode Mount(const size_t size, void*& buffer);

    // Unmount a segment. The buffer is the one returned by Mount.
    mooncake::ErrorCode Unmount(const void* buffer);

    mooncake::ErrorCode Get(const std::string& key, ggml_tensor * tensor);
    mooncake::ErrorCode Put(const std::string& key, const ggml_tensor * tensor);
    mooncake::ErrorCode Delete(const std::string& key);

    // Returns true if the key has a DISK replica
    // (master-assigned, written by PutToLocalFile).
    bool HasDiskReplica(const std::string& key);

    // Returns true if the key has a LOCAL_DISK replica
    // (created via the FileStorage offload path).
    bool HasLocalDiskReplica(const std::string& key);

    // Returns true if the key has a MEMORY replica.
    bool HasMemoryReplica(const std::string& key);

    // Like Get(), but takes the object size explicitly instead
    // of extracting it from a memory descriptor. This allows
    // reading from disk-only replicas where no memory
    // descriptor exists. Uses Client::Get(key, slices) which
    // handles routing to memory or disk internally.
    mooncake::ErrorCode GetWithExpectedSize(const std::string& key, size_t expected_size,
                                  std::string& value);

   private:
    struct SliceGuard {
        std::vector<mooncake::Slice> slices_;
        std::shared_ptr<mooncake::SimpleAllocator> allocator_;

        // Allocate memory according to the descriptors.
        SliceGuard(const std::vector<mooncake::AllocatedBuffer::Descriptor>& descriptors,
                   std::shared_ptr<mooncake::SimpleAllocator> allocator);

        // Allocate memory with a given size.
        SliceGuard(size_t size, std::shared_ptr<mooncake::SimpleAllocator> allocator);

        // Prevent copying
        SliceGuard(const SliceGuard&) = delete;
        SliceGuard& operator=(const SliceGuard&) = delete;

        ~SliceGuard();
    };

    // The client instance.
    std::shared_ptr<mooncake::Client> client_;
    // The segments that are mounted by the client.
    std::unordered_map<uintptr_t, MooncakeSegmentInfo> segments_;
    // Manage the memory allocation for get and put operations.
    std::shared_ptr<mooncake::SimpleAllocator> allocator_;
};


// TODO lhm_kv_cache_mooncake or lhm_memory_i?
class lhm_kv_cache_mooncake : public lhm_kv_cache {
   private:
    // actually we just have one client
    ClientInfo client_info;

    std::shared_ptr<MooncakeClientWrapper> GetClient() const {
        return client_info.client;
    }

    void HandleCreateClient(const std::string& hostname="localhost", const uint32_t port=4523);

    void HandlePut(const std::string& key, const ggml_tensor * tensor) const;

    void HandleGet(const std::string& key, ggml_tensor * tensor) const;

    void HandleMountClient(const std::string& segment_name, const size_t size);

    std::string GenerateKey(const std::string& tensor_name, const int32_t il, const std::string& type) const;

   public:
    lhm_kv_cache_mooncake(/* args */) = delete;
    ~lhm_kv_cache_mooncake() = default;

    // TODO: refactor the memory instances to not depend on `lhm_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct lhm_memory_params`
    lhm_kv_cache_mooncake(
            const lhm_model & model,
          const lhm_hparams & hparams,
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
               lhm_memory_t   mem_other,
        const lhm_memory_i::layer_filter_cb & filter,
        const  lhm_memory_i::layer_reuse_cb & reuse,
        const  lhm_memory_i::layer_share_cb & share);

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const lhm_kv_cache::slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const lhm_kv_cache::slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const lhm_kv_cache::slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const lhm_kv_cache::slot_info & sinfo) const;
};
