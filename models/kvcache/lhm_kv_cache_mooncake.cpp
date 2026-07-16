#include <cstring>
#include <stdexcept>

#include <utils.h>
#include <fmt/format.h>

#include "lhm_kv_cache_mooncake.h"
#include "log.h"
#include "config.h"


MooncakeClientWrapper::MooncakeClientWrapper(std::shared_ptr<mooncake::Client> client,
                                     std::shared_ptr<mooncake::SimpleAllocator> allocator)
    : client_(client), allocator_(allocator) {}

MooncakeClientWrapper::~MooncakeClientWrapper() {
    for (auto& [base, segment] : segments_) {
        free(segment.base);
    }
}

std::optional<std::shared_ptr<MooncakeClientWrapper>>
MooncakeClientWrapper::CreateClientWrapper(const std::string& hostname,
                                       const std::string& metadata_connstring,
                                       const std::string& protocol,
                                       const std::string& device_name,
                                       const std::string& master_server_entry,
                                       size_t local_buffer_size) {
    std::optional<std::string> device_names = std::nullopt;
    if (!device_name.empty()) {
        device_names = device_name;
    }

    auto client_opt = mooncake::Client::Create(hostname, metadata_connstring, protocol,
                                     device_names, master_server_entry);

    if (!client_opt.has_value()) {
        return std::nullopt;
    }

    std::shared_ptr<mooncake::SimpleAllocator> allocator =
        std::make_shared<mooncake::SimpleAllocator>(local_buffer_size);
    if (!allocator) {
        LOG_ERROR("Failed to create allocator");
        return std::nullopt;
    }

    // default register
    auto register_result = client_opt.value()->RegisterLocalMemory(
        allocator->getBase(), local_buffer_size, "cpu:0", false, false);
    mooncake::ErrorCode error_code =
        register_result.has_value() ? mooncake::ErrorCode::OK : register_result.error();
    if (error_code != mooncake::ErrorCode::OK) {
        LOG_ERROR("register_local_memory_failed base={} size={}, error={}", allocator->getBase(), local_buffer_size, mooncake::toString(error_code));
        return std::nullopt;
    }
    return std::make_shared<MooncakeClientWrapper>(client_opt.value(), allocator);
}

mooncake::ErrorCode MooncakeClientWrapper::Mount(const size_t size, void*& buffer) {
    buffer = mooncake::allocate_buffer_allocator_memory(size);
    if (!buffer) {
        LOG_ERROR("Failed to allocate memory for segment");
        return mooncake::ErrorCode::INTERNAL_ERROR;
    }

    auto mount_result = client_->MountSegment(buffer, size);
    mooncake::ErrorCode error_code =
        mount_result.has_value() ? mooncake::ErrorCode::OK : mount_result.error();
    if (error_code != mooncake::ErrorCode::OK) {
        free(buffer);
        return error_code;
    } else {
        segments_.emplace(reinterpret_cast<uintptr_t>(buffer),
                          MooncakeSegmentInfo{buffer, size});
        return mooncake::ErrorCode::OK;
    }
}

mooncake::ErrorCode MooncakeClientWrapper::Unmount(const void* buffer) {
    auto it = segments_.find(reinterpret_cast<uintptr_t>(buffer));
    if (it == segments_.end()) {
        return mooncake::ErrorCode::INVALID_PARAMS;
    }
    MooncakeSegmentInfo& segment = it->second;
    auto unmount_result = client_->UnmountSegment(segment.base, segment.size);
    mooncake::ErrorCode error_code =
        unmount_result.has_value() ? mooncake::ErrorCode::OK : unmount_result.error();
    if (error_code != mooncake::ErrorCode::OK) {
        return error_code;
    } else {
        // Clear the memory, so any further read will get wrong data
        memset(segment.base, 0, segment.size);
        free(segment.base);
        segments_.erase(it);
        return mooncake::ErrorCode::OK;
    }
}

mooncake::ErrorCode MooncakeClientWrapper::Get(const std::string& key, ggml_tensor * tensor) {
    auto query_result = client_->Query(key);
    if (!query_result.has_value()) {
        return query_result.error();
    }

    const std::vector<mooncake::Replica::Descriptor>& replica_list =
        query_result.value().replicas;
    if (replica_list.empty()) {
        return mooncake::ErrorCode::OBJECT_NOT_FOUND;
    }

    // Create slices
    const mooncake::AllocatedBuffer::Descriptor& descriptor =
        replica_list[0].get_memory_descriptor().buffer_descriptor;
    SliceGuard slice_guard(descriptor.size_, allocator_);

    // Perform get operation
    auto get_result =
        client_->Get(key, query_result.value(), slice_guard.slices_);
    mooncake::ErrorCode error_code =
        get_result.has_value() ? mooncake::ErrorCode::OK : get_result.error();
    if (error_code != mooncake::ErrorCode::OK) {
        return error_code;
    }

    // Fill value
    std::string value;
    for (const auto& slice : slice_guard.slices_) {
        value.append(static_cast<const char*>(slice.ptr), slice.size);
    }
    tensor->data = (void*)value.data();
    return mooncake::ErrorCode::OK;
}

mooncake::ErrorCode MooncakeClientWrapper::Put(const std::string& key,
                                    const ggml_tensor * tensor) {
    size_t bytes = ggml_nbytes(tensor);
    // Create slices
    SliceGuard slice_guard(bytes, allocator_);
    size_t offset = 0;
    for (const auto& slice : slice_guard.slices_) {
        memcpy(slice.ptr, static_cast<const char*>(tensor->data) + offset, slice.size);
        offset += slice.size;
    }

    // Configure replication
    mooncake::ReplicateConfig config;
    config.replica_num = 1;

    // Perform put operation
    auto put_result = client_->Put(key, slice_guard.slices_, config);
    return put_result.has_value() ? mooncake::ErrorCode::OK : put_result.error();
}

mooncake::ErrorCode MooncakeClientWrapper::Delete(const std::string& key) {
    auto remove_result = client_->Remove(key);
    return remove_result.has_value() ? mooncake::ErrorCode::OK : remove_result.error();
}

bool MooncakeClientWrapper::HasDiskReplica(const std::string& key) {
    auto query_result = client_->Query(key);
    if (!query_result.has_value()) return false;
    for (const auto& replica : query_result.value().replicas) {
        if (replica.is_disk_replica()) return true;
    }
    return false;
}

bool MooncakeClientWrapper::HasLocalDiskReplica(const std::string& key) {
    auto query_result = client_->Query(key);
    if (!query_result.has_value()) return false;
    for (const auto& replica : query_result.value().replicas) {
        if (replica.is_local_disk_replica()) return true;
    }
    return false;
}

bool MooncakeClientWrapper::HasMemoryReplica(const std::string& key) {
    auto query_result = client_->Query(key);
    if (!query_result.has_value()) return false;
    for (const auto& replica : query_result.value().replicas) {
        if (replica.is_memory_replica()) return true;
    }
    return false;
}

mooncake::ErrorCode MooncakeClientWrapper::GetWithExpectedSize(const std::string& key,
                                                 size_t expected_size,
                                                 std::string& value) {
    SliceGuard slice_guard(expected_size, allocator_);
    auto get_result = client_->Get(key, slice_guard.slices_);
    mooncake::ErrorCode error_code =
        get_result.has_value() ? mooncake::ErrorCode::OK : get_result.error();
    if (error_code != mooncake::ErrorCode::OK) {
        return error_code;
    }
    value.clear();
    for (const auto& slice : slice_guard.slices_) {
        value.append(static_cast<const char*>(slice.ptr), slice.size);
    }
    return mooncake::ErrorCode::OK;
}

MooncakeClientWrapper::SliceGuard::SliceGuard(
    const std::vector<mooncake::AllocatedBuffer::Descriptor>& descriptors,
    std::shared_ptr<mooncake::SimpleAllocator> allocator)
    : allocator_(allocator) {
    slices_.resize(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); i++) {
        void* buffer = allocator_->allocate(descriptors[i].size_);
        if (!buffer) {
            LOG_ERROR("Failed to allocate memory for slice");
            throw std::runtime_error("Failed to allocate memory for slice");
        }
        slices_[i] = mooncake::Slice{buffer, descriptors[i].size_};
    }
}

MooncakeClientWrapper::SliceGuard::SliceGuard(
    size_t size, std::shared_ptr<mooncake::SimpleAllocator> allocator)
    : allocator_(allocator) {
    while (size != 0) {
        auto chunk_size = std::min(size, mooncake::kMaxSliceSize);
        auto ptr = allocator_->allocate(chunk_size);
        if (!ptr) {
            LOG_ERROR("Failed to allocate memory for slice");
            throw std::runtime_error("Failed to allocate memory for slice");
        }
        slices_.emplace_back(mooncake::Slice{ptr, chunk_size});
        size -= chunk_size;
    }
}

MooncakeClientWrapper::SliceGuard::~SliceGuard() {
    for (auto& slice : slices_) {
        allocator_->deallocate(slice.ptr, slice.size);
    }
}

//
// lhm_kv_cache_mooncake
//

void lhm_kv_cache_mooncake::HandleCreateClient(const std::string& host, const uint32_t port) {
    if (host.empty() || port == 0) {
        LOG_ERROR("Invalid create command format. Expected: create [host:{}] [port:{}]", host, port);
        return;
    }

    std::string hostname = fmt::format("{}:{}", host, port);
    auto client_opt = MooncakeClientWrapper::CreateClientWrapper(
        hostname, FLAGS_mooncake_engine_meta_url, FLAGS_mooncake_protocol,
        FLAGS_mooncake_device_name, FLAGS_mooncake_master_server_entry);
    if (!client_opt.has_value()) {
        LOG_ERROR("Failed to create client: {}", hostname);
        return;
    }

    client_info = ClientInfo{client_opt.value(), {}, hostname};
    LOG_INFO("Successfully created client: {}", hostname);
}

void lhm_kv_cache_mooncake::HandlePut(const std::string& key, const ggml_tensor * tensor) const {
    if (key.empty() || !tensor->data) {
        LOG_WARN("Empty key or data");
        return;
    }

    mooncake::ErrorCode error_code = GetClient()->Put(key, tensor);
    if (error_code != mooncake::ErrorCode::OK) {
        LOG_ERROR("Failed to put data error code: {}", mooncake::toString(error_code));
        return;
    }
}

void lhm_kv_cache_mooncake::HandleGet(const std::string& key, ggml_tensor * tensor) const {
    if (key.empty()) {
        LOG_WARN("Empty key");
        return;
    }

    mooncake::ErrorCode error_code = GetClient()->Get(key, tensor);
    if (error_code != mooncake::ErrorCode::OK) {
        LOG_ERROR("Failed to get data error code: {}", mooncake::toString(error_code));
        return;
    }
}

void lhm_kv_cache_mooncake::HandleMountClient(const std::string& segment_name, const size_t size) {
    if (segment_name.empty() || size == 0) {
        LOG_WARN("Invalid mount command format. Expected: mount [segment_name:{}] [size:{}]", segment_name, size);
        return;
    }

    if (client_info.segments.find(segment_name) !=
        client_info.segments.end()) {
        LOG_WARN("Segment {} already mounted", segment_name);
        return;
    }

    void* base;
    mooncake::ErrorCode error_code = GetClient()->Mount(size, base);
    if (error_code != mooncake::ErrorCode::OK) {
        LOG_WARN("Failed to mount segment error code: {}", mooncake::toString(error_code));
        return;
    }

    client_info.segments[segment_name] = base;
}

std::string lhm_kv_cache_mooncake::GenerateKey(const std::string& tensor_name, const int32_t il, const std::string& type) const {
    return fmt::format("{}_{}_{}", tensor_name, il, type);
}

lhm_kv_cache_mooncake::lhm_kv_cache_mooncake(
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
    const  lhm_memory_i::layer_share_cb & share) :
    lhm_kv_cache(model, hparams, type_k, type_v, v_trans, offload, unified, 
        kv_size, n_seq_max, n_pad, n_swa, swa_type, mem_other, filter, reuse, share) {

    // use default
    HandleCreateClient();
}

ggml_tensor * lhm_kv_cache_mooncake::get_k(ggml_context * ctx, int32_t il,
                                            uint32_t n_kv, const lhm_kv_cache::slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    // data load from mooncake
    ggml_tensor * k = layers[ikv].k;
    std::string key = GenerateKey(k->name, il, "k");
    HandleGet(key, k);
    LHM_ASSERT(k->data);
    LOG_TRACE("get_k key:{} data:{}", key, k->data);

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * lhm_kv_cache_mooncake::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv,
                                            const lhm_kv_cache::slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    // data load from mooncake
    ggml_tensor * v = layers[ikv].v;
    std::string key = GenerateKey(v->name, il, "v");
    HandleGet(key, v);
    LHM_ASSERT(v->data);
    LOG_TRACE("get_v key:{} data:{}", key, v->data);

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * lhm_kv_cache_mooncake::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs,
                                            int32_t il, const lhm_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;
    std::string key = GenerateKey(k->name, il, "k");
    
    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    LHM_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    // TODO maybe wrong?
    ggml_tensor * result = ggml_set_rows(ctx, k, k_cur, k_idxs);
    LOG_TRACE("cpy_k key:{} data:{}", key, result->data);
    HandlePut(key, result);
    return result;
}

ggml_tensor * lhm_kv_cache_mooncake::cpy_v(ggml_context * ctx, ggml_tensor * v_cur,
                                        ggml_tensor * v_idxs, int32_t il,
                                        const lhm_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * v = layers[ikv].v;
    std::string key = GenerateKey(v->name, il, "v");

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    LHM_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        ggml_tensor * result = ggml_set_rows(ctx, v, v_cur, v_idxs);
        LOG_TRACE("cpy_v key:{} data:{}", key, result->data);
        HandlePut(key, result);
        return result;
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    ggml_tensor * result = ggml_set_rows(ctx, v_view, v_cur, v_idxs);
    LOG_TRACE("cpy_v key:{} data:{}", key, result->data);
    HandlePut(key, result);
    return result;
}
