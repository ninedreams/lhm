#include "ref_object.h"

namespace lhm {

ref_object::~ref_object()
{
  assert(nref == 0);
}

void ref_object::put() const {
  auto v = --nref;

  if (v == 0) {
    delete this;
  }
}

void ref_object::_get() const {
  auto v = ++nref;
  assert(v > 1); /* it should never happen that _get() sees nref == 0 */
}

}
