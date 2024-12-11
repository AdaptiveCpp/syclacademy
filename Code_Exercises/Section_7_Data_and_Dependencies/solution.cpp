/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include "../helpers.hpp"

#include <sycl/sycl.hpp>

int usm_selector(const sycl::device& dev) {
  if (dev.has(sycl::aspect::usm_device_allocations)) {
    if (dev.has(sycl::aspect::gpu))
      return 2;
    return 1;
  }
  return -1;
}

int main() {
  constexpr size_t dataSize = 1024;

  int inA[dataSize], inB[dataSize], inC[dataSize], out[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    inA[i] = static_cast<float>(i);
    inB[i] = static_cast<float>(i);
    inC[i] = static_cast<float>(i);
    out[i] = 0.0f;
  }

  try {
    auto usmQueue = sycl::queue { usm_selector };

    auto devicePtrInA = sycl::malloc_device<float>(dataSize, usmQueue);
    auto devicePtrInB = sycl::malloc_device<float>(dataSize, usmQueue);
    auto devicePtrInC = sycl::malloc_device<float>(dataSize, usmQueue);
    auto devicePtrOut = sycl::malloc_device<float>(dataSize, usmQueue);

    auto e1 = usmQueue.memcpy(devicePtrInA, inA, sizeof(float) * dataSize);
    auto e2 = usmQueue.memcpy(devicePtrInB, inB, sizeof(float) * dataSize);
    auto e3 = usmQueue.memcpy(devicePtrInC, inC, sizeof(float) * dataSize);

    auto e4 = usmQueue.parallel_for(
        sycl::range { dataSize }, e1, [=](sycl::id<1> idx) {
          auto globalId = idx[0];
          devicePtrInA[globalId] = devicePtrInA[globalId] * 2.0f;
        });

    auto e5 = usmQueue.parallel_for(
        sycl::range { dataSize }, { e2, e4 }, [=](sycl::id<1> idx) {
          auto globalId = idx[0];
          devicePtrInB[globalId] += devicePtrInA[globalId];
        });

    auto e6 = usmQueue.parallel_for(
        sycl::range { dataSize }, { e3, e4 }, [=](sycl::id<1> idx) {
          auto globalId = idx[0];
          devicePtrInC[globalId] -= devicePtrInA[globalId];
        });

    auto e7 = usmQueue.parallel_for(
        sycl::range { dataSize }, { e5, e6 }, [=](sycl::id<1> idx) {
          auto globalId = idx[0];
          devicePtrOut[globalId] =
              devicePtrInB[globalId] + devicePtrInC[globalId];
        });

    auto e8 = usmQueue.memcpy(out, devicePtrOut, sizeof(float) * dataSize, e7);

    e8.wait();

    sycl::free(devicePtrInA, usmQueue);
    sycl::free(devicePtrInB, usmQueue);
    sycl::free(devicePtrInC, usmQueue);
    sycl::free(devicePtrOut, usmQueue);

    usmQueue.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    SYCLACADEMY_ASSERT(out[i] == i * 2.0f);
  }
}
