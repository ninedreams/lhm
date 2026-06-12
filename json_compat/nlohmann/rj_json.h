#pragma once

// ============================================================================
// rj_json.h — Compatibility layer that wraps RapidJSON to provide an
// nlohmann/json-compatible API.  All existing code that uses
//   #include <nlohmann/json.hpp>   or   #include "nlohmann/json.hpp"
//   #include <nlohmann/json_fwd.hpp>
// will be redirected to this header via forwarding stubs, so no source files
// need to be modified.
// ============================================================================

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/reader.h>
#include <rapidjson/error/en.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// ============================================================================
namespace nlohmann {

// ---- Exception types -------------------------------------------------------

class exception : public std::exception {
public:
    explicit exception(const std::string & msg) : msg_(msg) {}
    const char * what() const noexcept override { return msg_.c_str(); }
protected:
    std::string msg_;
};

namespace detail {

class type_error : public nlohmann::exception {
public:
    explicit type_error(const std::string & msg) : nlohmann::exception(msg) {}
};

class out_of_range : public nlohmann::exception {
public:
    explicit out_of_range(const std::string & msg) : nlohmann::exception(msg) {}
};

class invalid_iterator : public nlohmann::exception {
public:
    explicit invalid_iterator(const std::string & msg) : nlohmann::exception(msg) {}
};

class other_error : public nlohmann::exception {
public:
    explicit other_error(const std::string & msg) : nlohmann::exception(msg) {}
};

// Forward declarations — defined after ordered_json
template <typename IteratorType> class iteration_proxy_value;
template <typename IteratorType> class iteration_proxy;

} // namespace detail

// ---- Forward declaration ----------------------------------------------------

class ordered_json;

// ---- SAX interface (needed by json_partial.cpp) ----------------------------

template <typename BasicJsonType>
struct json_sax {
    using number_integer_t  = int64_t;
    using number_unsigned_t = uint64_t;
    using number_float_t    = double;
    using string_t          = std::string;
    using binary_t          = std::vector<uint8_t>;

    virtual ~json_sax() = default;

    virtual bool null() = 0;
    virtual bool boolean(bool val) = 0;
    virtual bool number_integer(number_integer_t val) = 0;
    virtual bool number_unsigned(number_unsigned_t val) = 0;
    virtual bool number_float(number_float_t val, const string_t & s) = 0;
    virtual bool string(string_t & val) = 0;
    virtual bool binary(binary_t & val) = 0;
    virtual bool start_object(std::size_t len) = 0;
    virtual bool end_object() = 0;
    virtual bool start_array(std::size_t len) = 0;
    virtual bool end_array() = 0;
    virtual bool key(string_t & val) = 0;

    virtual bool parse_error(std::size_t position,
                             const std::string & last_token,
                             const typename BasicJsonType::exception & ex) = 0;
};

// ============================================================================
// ordered_json — the main JSON type
// ============================================================================

class ordered_json {
public:
    using value_type        = ordered_json;
    using size_type         = std::size_t;
    using difference_type   = std::ptrdiff_t;
    using exception         = nlohmann::exception;
    using string_t          = std::string;
    using number_integer_t  = int64_t;
    using number_unsigned_t = uint64_t;
    using number_float_t    = double;
    using binary_t          = std::vector<uint8_t>;

private:
    enum class value_t : uint8_t {
        null_value, object_value, array_value, string_value,
        bool_value, number_integer_value, number_unsigned_value,
        number_float_value, binary_value, discarded_value
    };

    value_t type_ = value_t::null_value;
    std::vector<std::pair<std::string, ordered_json>> object_entries_;
    std::vector<ordered_json> array_entries_;
    std::string   string_val_;
    int64_t       int_val_       = 0;
    uint64_t      uint_val_      = 0;
    double        float_val_     = 0.0;
    bool          bool_val_      = false;
    binary_t      binary_val_;

    [[noreturn]] void throw_type_error(const std::string & msg) const { throw detail::type_error(msg); }
    [[noreturn]] void throw_out_of_range(const std::string & msg) const { throw detail::out_of_range(msg); }

    int find_object_index(const std::string & key) const {
        for (int i = 0; i < (int)object_entries_.size(); ++i)
            if (object_entries_[i].first == key) return i;
        return -1;
    }
    ordered_json * find_object_value(const std::string & key) {
        int idx = find_object_index(key);
        return idx >= 0 ? &object_entries_[idx].second : nullptr;
    }
    const ordered_json * find_object_value(const std::string & key) const {
        int idx = find_object_index(key);
        return idx >= 0 ? &object_entries_[idx].second : nullptr;
    }

public:
    // ========================================================================
    // Construction
    // ========================================================================

    ordered_json() = default;
    ordered_json(std::nullptr_t) : type_(value_t::null_value) {}
    ordered_json(bool val) : type_(value_t::bool_value), bool_val_(val) {}

    ordered_json(int val)     : type_(value_t::number_integer_value), int_val_(val) {}
    ordered_json(int64_t val) : type_(value_t::number_integer_value), int_val_(val) {}

    ordered_json(unsigned int val) : type_(value_t::number_unsigned_value), uint_val_(val) {}
    ordered_json(uint64_t val)     : type_(value_t::number_unsigned_value), uint_val_(val) {}

    ordered_json(double val) : type_(value_t::number_float_value), float_val_(val) {}
    ordered_json(float val)  : type_(value_t::number_float_value), float_val_(static_cast<double>(val)) {}

    ordered_json(const char * val)        : type_(value_t::string_value), string_val_(val) {}
    ordered_json(const std::string & val) : type_(value_t::string_value), string_val_(val) {}
    ordered_json(std::string && val)      : type_(value_t::string_value), string_val_(std::move(val)) {}

    // Initializer-list constructor:  json j = {{"key", val}, ...}
    ordered_json(std::initializer_list<ordered_json> init) {
        if (init.size() == 2 && init.begin()->is_string() && !init.begin()[1].is_array_of_init()) {
            type_ = value_t::object_value;
            object_entries_.emplace_back(init.begin()->string_val_, init.begin()[1]);
        } else {
            bool all_kv = true;
            for (auto & el : init) {
                if (!el.is_array_of_init() || el.array_entries_.size() != 2 || !el.array_entries_[0].is_string()) {
                    all_kv = false; break;
                }
            }
            if (all_kv && init.size() > 0) {
                type_ = value_t::object_value;
                for (auto & el : init)
                    object_entries_.emplace_back(std::move(el.array_entries_[0].string_val_),
                                                 std::move(el.array_entries_[1]));
            } else {
                type_ = value_t::array_value;
                for (auto & el : init) array_entries_.push_back(std::move(el));
            }
        }
    }

    bool is_array_of_init() const { return type_ == value_t::array_value; }

    // Copy / move
    ordered_json(const ordered_json & other)
        : type_(other.type_), object_entries_(other.object_entries_),
          array_entries_(other.array_entries_), string_val_(other.string_val_),
          int_val_(other.int_val_), uint_val_(other.uint_val_),
          float_val_(other.float_val_), bool_val_(other.bool_val_),
          binary_val_(other.binary_val_) {}

    ordered_json(ordered_json && other) noexcept
        : type_(other.type_), object_entries_(std::move(other.object_entries_)),
          array_entries_(std::move(other.array_entries_)), string_val_(std::move(other.string_val_)),
          int_val_(other.int_val_), uint_val_(other.uint_val_),
          float_val_(other.float_val_), bool_val_(other.bool_val_),
          binary_val_(std::move(other.binary_val_)) { other.type_ = value_t::null_value; }

    ordered_json & operator=(const ordered_json & other) {
        if (this != &other) {
            type_ = other.type_; object_entries_ = other.object_entries_;
            array_entries_ = other.array_entries_; string_val_ = other.string_val_;
            int_val_ = other.int_val_; uint_val_ = other.uint_val_;
            float_val_ = other.float_val_; bool_val_ = other.bool_val_;
            binary_val_ = other.binary_val_;
        }
        return *this;
    }

    ordered_json & operator=(ordered_json && other) noexcept {
        if (this != &other) {
            type_ = other.type_; object_entries_ = std::move(other.object_entries_);
            array_entries_ = std::move(other.array_entries_); string_val_ = std::move(other.string_val_);
            int_val_ = other.int_val_; uint_val_ = other.uint_val_;
            float_val_ = other.float_val_; bool_val_ = other.bool_val_;
            binary_val_ = std::move(other.binary_val_); other.type_ = value_t::null_value;
        }
        return *this;
    }

    ~ordered_json() = default;

    // ========================================================================
    // Static constructors
    // ========================================================================

    static ordered_json object() { ordered_json j; j.type_ = value_t::object_value; return j; }
    static ordered_json array()  { ordered_json j; j.type_ = value_t::array_value;  return j; }

    static ordered_json array(std::initializer_list<ordered_json> init) {
        ordered_json j; j.type_ = value_t::array_value;
        for (auto & el : init) j.array_entries_.push_back(el);
        return j;
    }

    // ========================================================================
    // Type checking
    // ========================================================================

    bool is_null()             const { return type_ == value_t::null_value; }
    bool is_object()           const { return type_ == value_t::object_value; }
    bool is_array()            const { return type_ == value_t::array_value; }
    bool is_string()           const { return type_ == value_t::string_value; }
    bool is_boolean()          const { return type_ == value_t::bool_value; }
    bool is_number_integer()   const { return type_ == value_t::number_integer_value; }
    bool is_number_unsigned()  const { return type_ == value_t::number_unsigned_value; }
    bool is_number_float()     const { return type_ == value_t::number_float_value; }
    bool is_binary()           const { return type_ == value_t::binary_value; }
    bool is_discarded()        const { return type_ == value_t::discarded_value; }
    bool is_number() const { return is_number_integer() || is_number_unsigned() || is_number_float(); }
    bool is_primitive() const { return is_null() || is_string() || is_boolean() || is_number() || is_binary(); }
    bool is_structured() const { return is_object() || is_array(); }

    // ========================================================================
    // type_name()
    // ========================================================================

    const char * type_name() const {
        switch (type_) {
            case value_t::null_value:              return "null";
            case value_t::object_value:            return "object";
            case value_t::array_value:             return "array";
            case value_t::string_value:            return "string";
            case value_t::bool_value:              return "boolean";
            case value_t::number_integer_value:    return "number";
            case value_t::number_unsigned_value:   return "number";
            case value_t::number_float_value:      return "number";
            case value_t::binary_value:            return "binary";
            case value_t::discarded_value:         return "discarded";
        }
        return "unknown";
    }

    // ========================================================================
    // operator[]
    // ========================================================================

    const ordered_json & operator[](const std::string & key) const {
        if (type_ == value_t::null_value) { static const ordered_json null_static; return null_static; }
        if (type_ != value_t::object_value) throw_type_error("operator[] on non-object: " + std::string(type_name()));
        auto * v = find_object_value(key);
        if (!v) { static const ordered_json null_static; return null_static; }
        return *v;
    }

    ordered_json & operator[](const std::string & key) {
        if (type_ == value_t::null_value) type_ = value_t::object_value;
        if (type_ != value_t::object_value) throw_type_error("operator[] on non-object: " + std::string(type_name()));
        auto * v = find_object_value(key);
        if (!v) { object_entries_.emplace_back(key, ordered_json()); return object_entries_.back().second; }
        return *v;
    }

    const ordered_json & operator[](const char * key) const { return (*this)[std::string(key)]; }
    ordered_json & operator[](const char * key) { return (*this)[std::string(key)]; }

    const ordered_json & operator[](std::size_t idx) const {
        if (type_ != value_t::array_value) throw_type_error("operator[] idx on non-array: " + std::string(type_name()));
        if (idx >= array_entries_.size()) throw_out_of_range("array index out of range");
        return array_entries_[idx];
    }
    ordered_json & operator[](std::size_t idx) {
        if (type_ != value_t::array_value) throw_type_error("operator[] idx on non-array: " + std::string(type_name()));
        if (idx >= array_entries_.size()) throw_out_of_range("array index out of range");
        return array_entries_[idx];
    }

    // ========================================================================
    // at()
    // ========================================================================

    const ordered_json & at(const std::string & key) const {
        if (type_ != value_t::object_value) throw_type_error("at() on non-object: " + std::string(type_name()));
        auto * v = find_object_value(key);
        if (!v) throw_out_of_range("key not found: " + key);
        return *v;
    }
    ordered_json & at(const std::string & key) {
        if (type_ != value_t::object_value) throw_type_error("at() on non-object: " + std::string(type_name()));
        auto * v = find_object_value(key);
        if (!v) throw_out_of_range("key not found: " + key);
        return *v;
    }
    const ordered_json & at(std::size_t idx) const {
        if (type_ != value_t::array_value) throw_type_error("at() idx on non-array: " + std::string(type_name()));
        if (idx >= array_entries_.size()) throw_out_of_range("array index out of range");
        return array_entries_[idx];
    }
    ordered_json & at(std::size_t idx) {
        if (type_ != value_t::array_value) throw_type_error("at() idx on non-array: " + std::string(type_name()));
        if (idx >= array_entries_.size()) throw_out_of_range("array index out of range");
        return array_entries_[idx];
    }

    // ========================================================================
    // contains() / count()
    // ========================================================================

    bool contains(const std::string & key) const { return type_ == value_t::object_value && find_object_index(key) >= 0; }
    bool contains(const char * key) const { return contains(std::string(key)); }
    std::size_t count(const std::string & key) const { return contains(key) ? 1 : 0; }

    // ========================================================================
    // get<T>() / get_to<T>()
    // ========================================================================

    template <typename T>
    T get() const {
        if constexpr (std::is_same_v<T, bool>) {
            if (!is_boolean()) throw_type_error("get<bool>() on " + std::string(type_name()));
            return bool_val_;
        } else if constexpr (std::is_same_v<T, int>) {
            if (is_number_integer()) return static_cast<int>(int_val_);
            if (is_number_unsigned()) return static_cast<int>(uint_val_);
            if (is_number_float()) return static_cast<int>(float_val_);
            throw_type_error("get<int>() on " + std::string(type_name()));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (is_number_integer()) return int_val_;
            if (is_number_unsigned()) return static_cast<int64_t>(uint_val_);
            if (is_number_float()) return static_cast<int64_t>(float_val_);
            throw_type_error("get<int64_t>() on " + std::string(type_name()));
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            if (is_number_unsigned()) return uint_val_;
            if (is_number_integer()) return static_cast<uint64_t>(int_val_);
            if (is_number_float()) return static_cast<uint64_t>(float_val_);
            throw_type_error("get<uint64_t>() on " + std::string(type_name()));
        } else if constexpr (std::is_same_v<T, double>) {
            if (is_number_float()) return float_val_;
            if (is_number_integer()) return static_cast<double>(int_val_);
            if (is_number_unsigned()) return static_cast<double>(uint_val_);
            throw_type_error("get<double>() on " + std::string(type_name()));
        } else if constexpr (std::is_same_v<T, float>) {
            if (is_number_float()) return static_cast<float>(float_val_);
            if (is_number_integer()) return static_cast<float>(static_cast<double>(int_val_));
            if (is_number_unsigned()) return static_cast<float>(static_cast<double>(uint_val_));
            throw_type_error("get<float>() on " + std::string(type_name()));
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (!is_string()) throw_type_error("get<string>() on " + std::string(type_name()));
            return string_val_;
        } else if constexpr (std::is_same_v<T, ordered_json>) {
            return *this;
        } else if constexpr (std::is_enum_v<T>) {
            if (is_number_integer()) return static_cast<T>(int_val_);
            if (is_number_unsigned()) return static_cast<T>(uint_val_);
            throw_type_error("get<enum>() on " + std::string(type_name()));
        } else {
            static_assert(!std::is_same_v<T, T>, "get<T>: unsupported type");
        }
        return T{};
    }

    // get_to — assigns the value to the output parameter
    template <typename T>
    void get_to(T & val) const { val = get<T>(); }

    // get_to specialization for std::set<std::string>
    void get_to(std::set<std::string> & val) const {
        if (!is_array()) throw_type_error("get_to<set>() on non-array: " + std::string(type_name()));
        val.clear();
        for (const auto & item : array_entries_) val.insert(item.get<std::string>());
    }

    // get_to specialization for std::vector<std::string>
    void get_to(std::vector<std::string> & val) const {
        if (!is_array()) throw_type_error("get_to<vector>() on non-array: " + std::string(type_name()));
        val.clear();
        for (const auto & item : array_entries_) val.push_back(item.get<std::string>());
    }

    // get_to for std::vector<int>
    void get_to(std::vector<int> & val) const {
        if (!is_array()) throw_type_error("get_to<vector<int>>() on non-array: " + std::string(type_name()));
        val.clear();
        for (const auto & item : array_entries_) val.push_back(item.get<int>());
    }

    // ========================================================================
    // Implicit conversion operators
    // ========================================================================

    operator bool()              const { return get<bool>(); }
    operator int()               const { return get<int>(); }
    operator int64_t()           const { return get<int64_t>(); }
    operator uint64_t()          const { return get<uint64_t>(); }
    operator double()            const { return get<double>(); }
    operator std::string()       const { return get<std::string>(); }

    // ========================================================================
    // value() — get with default
    // ========================================================================

    template <typename T>
    T value(const std::string & key, const T & default_val) const {
        if (!contains(key)) return default_val;
        try { return at(key).get<T>(); } catch (...) { return default_val; }
    }

    // ========================================================================
    // size() / empty() / clear()
    // ========================================================================

    std::size_t size() const {
        switch (type_) {
            case value_t::object_value: return object_entries_.size();
            case value_t::array_value:  return array_entries_.size();
            case value_t::null_value:   return 0;
            default:                    return 1;
        }
    }

    bool empty() const {
        switch (type_) {
            case value_t::object_value: return object_entries_.empty();
            case value_t::array_value:  return array_entries_.empty();
            case value_t::null_value:   return true;
            case value_t::string_value: return string_val_.empty();
            default:                    return false;
        }
    }

    void clear() {
        type_ = value_t::null_value;
        object_entries_.clear(); array_entries_.clear();
        string_val_.clear(); binary_val_.clear();
    }

    // ========================================================================
    // push_back()
    // ========================================================================

    void push_back(const ordered_json & val) {
        if (type_ == value_t::null_value) type_ = value_t::array_value;
        if (type_ != value_t::array_value) throw_type_error("push_back on non-array: " + std::string(type_name()));
        array_entries_.push_back(val);
    }
    void push_back(ordered_json && val) {
        if (type_ == value_t::null_value) type_ = value_t::array_value;
        if (type_ != value_t::array_value) throw_type_error("push_back on non-array: " + std::string(type_name()));
        array_entries_.push_back(std::move(val));
    }
    void push_back(const std::pair<std::string, ordered_json> & p) {
        if (type_ == value_t::null_value) type_ = value_t::object_value;
        if (type_ != value_t::object_value) throw_type_error("push_back(pair) on non-object: " + std::string(type_name()));
        int idx = find_object_index(p.first);
        if (idx >= 0) object_entries_[idx].second = p.second;
        else object_entries_.push_back(p);
    }

    // ========================================================================
    // emplace()
    // ========================================================================

    std::pair<ordered_json *, bool> emplace(const std::string & key, ordered_json && val) {
        if (type_ == value_t::null_value) type_ = value_t::object_value;
        int idx = find_object_index(key);
        if (idx >= 0) return {&object_entries_[idx].second, false};
        object_entries_.emplace_back(key, std::move(val));
        return {&object_entries_.back().second, true};
    }

    // ========================================================================
    // erase()
    // ========================================================================

    void erase(const std::string & key) {
        if (type_ != value_t::object_value) throw_type_error("erase(key) on non-object: " + std::string(type_name()));
        int idx = find_object_index(key);
        if (idx >= 0) object_entries_.erase(object_entries_.begin() + idx);
    }

    // ========================================================================
    // update()
    // ========================================================================

    void update(const ordered_json & other) {
        if (type_ == value_t::null_value) type_ = value_t::object_value;
        if (type_ != value_t::object_value) throw_type_error("update() on non-object");
        if (other.is_object()) {
            for (auto & [k, v] : other.object_entries_) {
                int idx = find_object_index(k);
                if (idx >= 0) object_entries_[idx].second = v;
                else object_entries_.emplace_back(k, v);
            }
        }
    }

    // ========================================================================
    // dump() — serialization
    // ========================================================================

    std::string dump(int indent = -1, char indent_char = ' ') const {
        std::string result;
        dump_impl(result, indent, indent_char, 0);
        return result;
    }

private:
    void dump_impl(std::string & out, int indent, char indent_char, int depth) const {
        switch (type_) {
            case value_t::null_value:    out += "null"; break;
            case value_t::bool_value:    out += bool_val_ ? "true" : "false"; break;
            case value_t::number_integer_value:  out += std::to_string(int_val_); break;
            case value_t::number_unsigned_value: out += std::to_string(uint_val_); break;
            case value_t::number_float_value: {
                rapidjson::StringBuffer sb;
                rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
                writer.Double(float_val_);
                out += sb.GetString();
                break;
            }
            case value_t::string_value: dump_string(out, string_val_); break;
            case value_t::array_value:
                out += '[';
                if (indent >= 0 && !array_entries_.empty()) {
                    out += '\n';
                    for (std::size_t i = 0; i < array_entries_.size(); ++i) {
                        out.append((depth + 1) * indent, indent_char);
                        array_entries_[i].dump_impl(out, indent, indent_char, depth + 1);
                        if (i + 1 < array_entries_.size()) out += ',';
                        out += '\n';
                    }
                    out.append(depth * indent, indent_char);
                } else {
                    for (std::size_t i = 0; i < array_entries_.size(); ++i) {
                        if (i > 0) out += ',';
                        array_entries_[i].dump_impl(out, indent, indent_char, depth + 1);
                    }
                }
                out += ']';
                break;
            case value_t::object_value:
                out += '{';
                if (indent >= 0 && !object_entries_.empty()) {
                    out += '\n';
                    for (std::size_t i = 0; i < object_entries_.size(); ++i) {
                        out.append((depth + 1) * indent, indent_char);
                        dump_string(out, object_entries_[i].first);
                        out += ':';
                        if (indent >= 0) out += ' ';
                        object_entries_[i].second.dump_impl(out, indent, indent_char, depth + 1);
                        if (i + 1 < object_entries_.size()) out += ',';
                        out += '\n';
                    }
                    out.append(depth * indent, indent_char);
                } else {
                    for (std::size_t i = 0; i < object_entries_.size(); ++i) {
                        if (i > 0) out += ',';
                        dump_string(out, object_entries_[i].first);
                        out += ':';
                        object_entries_[i].second.dump_impl(out, indent, indent_char, depth + 1);
                    }
                }
                out += '}';
                break;
            case value_t::binary_value:   out += "null"; break;
            case value_t::discarded_value: break;
        }
    }

    static void dump_string(std::string & out, const std::string & s) {
        out += '"';
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        out += buf;
                    } else {
                        out += c;
                    }
                    break;
            }
        }
        out += '"';
    }

public:
    // ========================================================================
    // Parsing
    // ========================================================================

    static ordered_json parse(const std::string & str) {
        return parse(str.data(), str.data() + str.size());
    }
    static ordered_json parse(const char * str, std::size_t len) {
        return parse(str, str + len);
    }
    template <typename InputIt>
    static ordered_json parse(InputIt first, InputIt last) {
        std::string str(first, last);
        return parse(str.data(), str.data() + str.size());
    }
    static ordered_json parse(const char * begin, const char * end) {
        rapidjson::Document doc;
        doc.Parse(begin, static_cast<std::size_t>(end - begin));
        if (doc.HasParseError()) {
            throw exception(std::string("JSON parse error: ") +
                            rapidjson::GetParseError_En(doc.GetParseError()) +
                            " at offset " + std::to_string(doc.GetErrorOffset()));
        }
        return from_rapidjson(doc);
    }

private:
    static ordered_json from_rapidjson(const rapidjson::Value & v) {
        switch (v.GetType()) {
            case rapidjson::kNullType:   return ordered_json(nullptr);
            case rapidjson::kFalseType:  return ordered_json(false);
            case rapidjson::kTrueType:   return ordered_json(true);
            case rapidjson::kObjectType: {
                ordered_json j = ordered_json::object();
                for (auto it = v.MemberBegin(); it != v.MemberEnd(); ++it) {
                    std::string key(it->name.GetString(), it->name.GetStringLength());
                    j.object_entries_.emplace_back(std::move(key), from_rapidjson(it->value));
                }
                return j;
            }
            case rapidjson::kArrayType: {
                ordered_json j = ordered_json::array();
                for (auto it = v.Begin(); it != v.End(); ++it)
                    j.array_entries_.push_back(from_rapidjson(*it));
                return j;
            }
            case rapidjson::kStringType:
                return ordered_json(std::string(v.GetString(), v.GetStringLength()));
            case rapidjson::kNumberType:
                if (v.IsInt64())  return ordered_json(v.GetInt64());
                if (v.IsUint64()) return ordered_json(v.GetUint64());
                if (v.IsInt())    return ordered_json(static_cast<int64_t>(v.GetInt()));
                if (v.IsUint())   return ordered_json(static_cast<uint64_t>(v.GetUint()));
                return ordered_json(v.GetDouble());
            default:
                return ordered_json(nullptr);
        }
    }

    // ---- SAX handler (defined before sax_parse methods) ---------------------

    class sax_handler {
    public:
        sax_handler(json_sax<ordered_json> * sax) : sax_(sax), ok_(true) {}

        bool Default() { return ok_; }
        bool Null()       { if (!ok_) return false; ok_ = sax_->null();              return ok_; }
        bool Bool(bool b) { if (!ok_) return false; ok_ = sax_->boolean(b);         return ok_; }
        bool Int(int i)   { if (!ok_) return false; ok_ = sax_->number_integer(static_cast<int64_t>(i));  return ok_; }
        bool Uint(unsigned u) { if (!ok_) return false; ok_ = sax_->number_unsigned(static_cast<uint64_t>(u)); return ok_; }
        bool Int64(int64_t i) { if (!ok_) return false; ok_ = sax_->number_integer(i);  return ok_; }
        bool Uint64(uint64_t u) { if (!ok_) return false; ok_ = sax_->number_unsigned(u); return ok_; }
        bool Double(double d) { if (!ok_) return false; std::string s = std::to_string(d); ok_ = sax_->number_float(d, s); return ok_; }
        bool String(const char * str, rapidjson::SizeType length, bool) {
            if (!ok_) return false; std::string s(str, length); ok_ = sax_->string(s); return ok_;
        }
        bool StartObject() { if (!ok_) return false; ok_ = sax_->start_object(static_cast<std::size_t>(-1)); return ok_; }
        bool EndObject(rapidjson::SizeType) { if (!ok_) return false; ok_ = sax_->end_object(); return ok_; }
        bool Key(const char * str, rapidjson::SizeType length, bool) {
            if (!ok_) return false; std::string s(str, length); ok_ = sax_->key(s); return ok_;
        }
        bool StartArray() { if (!ok_) return false; ok_ = sax_->start_array(static_cast<std::size_t>(-1)); return ok_; }
        bool EndArray(rapidjson::SizeType) { if (!ok_) return false; ok_ = sax_->end_array(); return ok_; }
        bool RawNumber(const char *, rapidjson::SizeType, bool) { return true; }

        bool HasError() const { return !ok_; }

    private:
        json_sax<ordered_json> * sax_;
        bool ok_;
    };

public:
    // ========================================================================
    // SAX parsing
    // ========================================================================

    template <typename InputIt>
    static bool sax_parse(InputIt & first, InputIt last, json_sax<ordered_json> * sax) {
        std::string str(first, last);
        bool result = sax_parse_internal(str, sax);
        if (result) first = last;
        return result;
    }

    static bool sax_parse(const std::string & str, json_sax<ordered_json> * sax) {
        return sax_parse_internal(str, sax);
    }

    // Overload with two iterator references (for json_partial.cpp)
    template <typename InputIt>
    static bool sax_parse(InputIt & first, const InputIt & last, json_sax<ordered_json> * sax) {
        std::string str(first, last);
        rapidjson::Reader reader;
        rapidjson::StringStream ss(str.c_str());
        sax_handler handler(sax);
        rapidjson::ParseResult result = reader.Parse<rapidjson::kParseDefaultFlags>(ss, handler);
        if (!result) {
            std::size_t offset = result.Offset();
            std::string last_token;
            exception ex(std::string("JSON parse error at offset ") + std::to_string(offset));
            sax->parse_error(offset, last_token, ex);
            return false;
        }
        return true;
    }

private:
    static bool sax_parse_internal(const std::string & str, json_sax<ordered_json> * sax) {
        rapidjson::Reader reader;
        rapidjson::StringStream ss(str.c_str());
        sax_handler handler(sax);
        rapidjson::ParseResult result = reader.Parse<rapidjson::kParseDefaultFlags>(ss, handler);
        if (!result) {
            std::size_t offset = result.Offset();
            std::string last_token;
            exception ex(std::string("JSON parse error at offset ") + std::to_string(offset));
            sax->parse_error(offset, last_token, ex);
            return false;
        }
        return true;
    }

public:
    // ========================================================================
    // Iteration — iterator / const_iterator
    // ========================================================================

    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = ordered_json;
        using difference_type   = std::ptrdiff_t;
        using pointer           = ordered_json *;
        using reference         = ordered_json &;

        enum class iter_type { object_iter, array_iter, null_iter };

    private:
        iter_type type_;
        std::vector<std::pair<std::string, ordered_json>> * obj_;
        std::size_t obj_idx_;
        std::vector<ordered_json> * arr_;
        std::size_t arr_idx_;
        friend class const_iterator;
        friend class ordered_json;

    public:
        iterator() : type_(iter_type::null_iter), obj_(nullptr), obj_idx_(0), arr_(nullptr), arr_idx_(0) {}
        iterator(std::vector<std::pair<std::string, ordered_json>> * obj, std::size_t idx)
            : type_(iter_type::object_iter), obj_(obj), obj_idx_(idx), arr_(nullptr), arr_idx_(0) {}
        iterator(std::vector<ordered_json> * arr, std::size_t idx)
            : type_(iter_type::array_iter), obj_(nullptr), obj_idx_(0), arr_(arr), arr_idx_(idx) {}

        std::string key() const {
            if (type_ != iter_type::object_iter) throw detail::invalid_iterator("key() on non-object iterator");
            return obj_->at(obj_idx_).first;
        }
        ordered_json & value() const {
            if (type_ == iter_type::object_iter) return obj_->at(obj_idx_).second;
            if (type_ == iter_type::array_iter)  return arr_->at(arr_idx_);
            throw detail::invalid_iterator("value() on invalid iterator");
        }
        ordered_json & operator*() const { return value(); }
        ordered_json * operator->() const { return &value(); }
        iterator & operator++() { if (type_ == iter_type::object_iter) ++obj_idx_; else if (type_ == iter_type::array_iter) ++arr_idx_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }
        bool operator==(const iterator & o) const {
            if (type_ != o.type_) return false;
            if (type_ == iter_type::object_iter) return obj_idx_ == o.obj_idx_ && obj_ == o.obj_;
            if (type_ == iter_type::array_iter)  return arr_idx_ == o.arr_idx_ && arr_ == o.arr_;
            return true;
        }
        bool operator!=(const iterator & o) const { return !(*this == o); }
    };

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = ordered_json;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const ordered_json *;
        using reference         = const ordered_json &;

        enum class iter_type { object_iter, array_iter, null_iter };

    private:
        iter_type type_;
        const std::vector<std::pair<std::string, ordered_json>> * obj_;
        std::size_t obj_idx_;
        const std::vector<ordered_json> * arr_;
        std::size_t arr_idx_;

    public:
        const_iterator() : type_(iter_type::null_iter), obj_(nullptr), obj_idx_(0), arr_(nullptr), arr_idx_(0) {}
        const_iterator(const std::vector<std::pair<std::string, ordered_json>> * obj, std::size_t idx)
            : type_(iter_type::object_iter), obj_(obj), obj_idx_(idx), arr_(nullptr), arr_idx_(0) {}
        const_iterator(const std::vector<ordered_json> * arr, std::size_t idx)
            : type_(iter_type::array_iter), obj_(nullptr), obj_idx_(0), arr_(arr), arr_idx_(idx) {}
        const_iterator(const iterator & it)
            : type_(iter_type(it.type_)), obj_(it.obj_), obj_idx_(it.obj_idx_), arr_(it.arr_), arr_idx_(it.arr_idx_) {}

        std::string key() const {
            if (type_ != iter_type::object_iter) throw detail::invalid_iterator("key() on non-object iterator");
            return obj_->at(obj_idx_).first;
        }
        const ordered_json & value() const {
            if (type_ == iter_type::object_iter) return obj_->at(obj_idx_).second;
            if (type_ == iter_type::array_iter)  return arr_->at(arr_idx_);
            throw detail::invalid_iterator("value() on invalid iterator");
        }
        const ordered_json & operator*() const { return value(); }
        const ordered_json * operator->() const { return &value(); }
        const_iterator & operator++() { if (type_ == iter_type::object_iter) ++obj_idx_; else if (type_ == iter_type::array_iter) ++arr_idx_; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++(*this); return tmp; }
        bool operator==(const const_iterator & o) const {
            if (type_ != o.type_) return false;
            if (type_ == iter_type::object_iter) return obj_idx_ == o.obj_idx_ && obj_ == o.obj_;
            if (type_ == iter_type::array_iter)  return arr_idx_ == o.arr_idx_ && arr_ == o.arr_;
            return true;
        }
        bool operator!=(const const_iterator & o) const { return !(*this == o); }
    };

    // ---- begin() / end() ---------------------------------------------------

    iterator begin() {
        if (type_ == value_t::object_value) return iterator(&object_entries_, 0);
        if (type_ == value_t::array_value)  return iterator(&array_entries_, 0);
        return iterator();
    }
    iterator end() {
        if (type_ == value_t::object_value) return iterator(&object_entries_, object_entries_.size());
        if (type_ == value_t::array_value)  return iterator(&array_entries_, array_entries_.size());
        return iterator();
    }
    const_iterator begin() const {
        if (type_ == value_t::object_value) return const_iterator(&object_entries_, 0);
        if (type_ == value_t::array_value)  return const_iterator(&array_entries_, 0);
        return const_iterator();
    }
    const_iterator end() const {
        if (type_ == value_t::object_value) return const_iterator(&object_entries_, object_entries_.size());
        if (type_ == value_t::array_value)  return const_iterator(&array_entries_, array_entries_.size());
        return const_iterator();
    }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend()   const { return end(); }

    // ---- items() — declared here, defined after detail types ----------------

    detail::iteration_proxy<iterator> items();
    detail::iteration_proxy<const_iterator> items() const;

    // ---- erase by iterator -------------------------------------------------

    iterator erase(iterator it) {
        if (it.type_ == iterator::iter_type::object_iter && it.obj_ == &object_entries_) {
            std::size_t idx = it.obj_idx_;
            object_entries_.erase(object_entries_.begin() + idx);
            return iterator(&object_entries_, idx);
        } else if (it.type_ == iterator::iter_type::array_iter && it.arr_ == &array_entries_) {
            std::size_t idx = it.arr_idx_;
            array_entries_.erase(array_entries_.begin() + idx);
            return iterator(&array_entries_, idx);
        }
        return end();
    }

    // ========================================================================
    // Comparison operators
    // ========================================================================

    bool operator==(const ordered_json & o) const {
        if (type_ != o.type_) return false;
        switch (type_) {
            case value_t::null_value:    return true;
            case value_t::bool_value:    return bool_val_ == o.bool_val_;
            case value_t::number_integer_value:  return int_val_ == o.int_val_;
            case value_t::number_unsigned_value: return uint_val_ == o.uint_val_;
            case value_t::number_float_value:    return float_val_ == o.float_val_;
            case value_t::string_value:  return string_val_ == o.string_val_;
            case value_t::object_value:  return object_entries_ == o.object_entries_;
            case value_t::array_value:   return array_entries_ == o.array_entries_;
            default: return false;
        }
    }
    bool operator!=(const ordered_json & o) const { return !(*this == o); }
    bool operator<(const ordered_json & o) const {
        if (type_ != o.type_) return static_cast<int>(type_) < static_cast<int>(o.type_);
        switch (type_) {
            case value_t::number_integer_value:  return int_val_ < o.int_val_;
            case value_t::number_unsigned_value: return uint_val_ < o.uint_val_;
            case value_t::number_float_value:    return float_val_ < o.float_val_;
            case value_t::string_value:          return string_val_ < o.string_val_;
            default: return false;
        }
    }
    bool operator<=(const ordered_json & o) const { return !(o < *this); }
    bool operator>(const ordered_json & o) const  { return o < *this; }
    bool operator>=(const ordered_json & o) const { return !(*this < o); }

    // ========================================================================
    // Swap / front / back
    // ========================================================================

    void swap(ordered_json & o) noexcept {
        std::swap(type_, o.type_); object_entries_.swap(o.object_entries_);
        array_entries_.swap(o.array_entries_); string_val_.swap(o.string_val_);
        std::swap(int_val_, o.int_val_); std::swap(uint_val_, o.uint_val_);
        std::swap(float_val_, o.float_val_); std::swap(bool_val_, o.bool_val_);
        binary_val_.swap(o.binary_val_);
    }

    ordered_json & front() {
        if (type_ == value_t::array_value && !array_entries_.empty()) return array_entries_.front();
        throw_type_error("front() on non-array or empty");
    }
    const ordered_json & front() const {
        if (type_ == value_t::array_value && !array_entries_.empty()) return array_entries_.front();
        throw_type_error("front() on non-array or empty");
    }
    ordered_json & back() {
        if (type_ == value_t::array_value && !array_entries_.empty()) return array_entries_.back();
        throw_type_error("back() on non-array or empty");
    }
    const ordered_json & back() const {
        if (type_ == value_t::array_value && !array_entries_.empty()) return array_entries_.back();
        throw_type_error("back() on non-array or empty");
    }

    // ========================================================================
    // to_json / from_json free functions
    // ========================================================================

    friend void to_json(ordered_json & j, const ordered_json & val) { j = val; }
    friend void from_json(const ordered_json & j, ordered_json & val) { val = j; }
};

// ============================================================================
// nlohmann::json — alias to ordered_json
// ============================================================================
using json = ordered_json;

// ============================================================================
// detail::iteration_proxy_value / iteration_proxy
// ============================================================================

namespace detail {

template <typename IteratorType>
class iteration_proxy_value {
public:
    // Structured binding support: first/second are the only public data members,
    // so C++17 decomposes into exactly 2 elements: auto& [k, v] = *it;
    // Note: second is a pointer (not reference) so it can be updated in operator++.
    typename IteratorType::pointer second;
    std::string first;

    iteration_proxy_value(IteratorType it)
        : first(it.key()), second(&it.value()), it_(it) {}

    std::string key() const { return it_.key(); }
    typename IteratorType::reference value() const { return it_.value(); }

    iteration_proxy_value & operator*() { return *this; }
    iteration_proxy_value & operator++() { ++it_; first = it_.key(); second = &it_.value(); return *this; }
    bool operator!=(const iteration_proxy_value & other) const { return it_ != other.it_; }

private:
    IteratorType it_;
};

} // namespace detail

namespace detail {

template <typename IteratorType>
class iteration_proxy {
public:
    iteration_proxy(IteratorType begin, IteratorType end) : begin_(begin), end_(end) {}

    iteration_proxy_value<IteratorType> begin() { return iteration_proxy_value<IteratorType>(begin_); }
    iteration_proxy_value<IteratorType> end()   { return iteration_proxy_value<IteratorType>(end_); }

private:
    IteratorType begin_;
    IteratorType end_;
};

} // namespace detail

// ============================================================================
// items() out-of-line definitions (depend on detail::iteration_proxy)
// ============================================================================

inline detail::iteration_proxy<ordered_json::iterator> ordered_json::items() {
    if (!is_object()) throw_type_error("items() on non-object: " + std::string(type_name()));
    return detail::iteration_proxy<iterator>(begin(), end());
}

inline detail::iteration_proxy<ordered_json::const_iterator> ordered_json::items() const {
    if (!is_object()) throw_type_error("items() on non-object: " + std::string(type_name()));
    return detail::iteration_proxy<const_iterator>(begin(), end());
}

} // namespace nlohmann

// ============================================================================
// NLOHMANN_JSON_NAMESPACE shim
// ============================================================================
namespace NLOHMANN_JSON_NAMESPACE = nlohmann;

// ============================================================================
// std::hash specialization
// ============================================================================
namespace std {
template <>
struct hash<nlohmann::ordered_json> {
    std::size_t operator()(const nlohmann::ordered_json & j) const noexcept {
        return std::hash<std::string>()(j.dump());
    }
};
} // namespace std
