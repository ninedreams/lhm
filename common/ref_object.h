#pragma once

#include <cassert>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace lhm {

class ref_object {
public:
  uint64_t get_nref() const {
    return nref;
  }

  const ref_object *get() const {
    _get();
    return this;
  }
  ref_object *get() {
    _get();
    return this;
  }
  void put() const;

protected:
  ref_object() = default;
  ref_object(const ref_object& o) : cct(o.cct) {}
  ref_object& operator=(const ref_object& o) = delete;
  ref_object(ref_object&&) = delete;
  ref_object& operator=(ref_object&&) = delete;
  ref_object() {}

  virtual ~ref_object();

private:
  void _get() const;

  mutable std::atomic<uint64_t> nref{1};
};


class ref_object_safe : public ref_object {
public:
  ref_object *get() = delete;
  const ref_object *get() const = delete;
  void put() const = delete;
protected:
template<typename... Args>
  ref_object_safe(Args&&... args) : ref_object(std::forward<Args>(args)...) {}
  virtual ~ref_object_safe() override {}
};


struct ref_object_cond : public ref_object {
  ref_object_cond() = default;
  ~ref_object_cond() = default;

  int wait() {
    std::unique_lock l(lock);
    while (!complete) {
      cond.wait(l);
    }
    return rval;
  }

  void done(int r) {
    std::lock_guard l(lock);
    rval = r;
    complete = true;
    cond.notify_all();
  }

  void done() {
    done(0);
  }

private:
  bool complete = false;
  std::mutex lock;
  std::condition_variable cond;
  int rval = 0;
};


struct ref_wait_object {
  std::atomic<uint64_t> nref = { 1 };
  ref_object_cond *c;

  ref_wait_object() {
    c = new ref_object_cond;
  }
  virtual ~ref_wait_object() {
    c->put();
  }

  ref_wait_object *get() {
    nref++;
    return this;
  }

  bool put() {
    bool ret = false;
    ref_object_cond *cond = c;
    cond->get();
    if (--nref == 0) {
      cond->done();
      delete this;
      ret = true;
    }
    cond->put();
    return ret;
  }

  void put_wait() {
    ref_object_cond *cond = c;

    cond->get();
    if (--nref == 0) {
      cond->done();
      delete this;
    } else {
      cond->wait();
    }
    cond->put();
  }
};

}
