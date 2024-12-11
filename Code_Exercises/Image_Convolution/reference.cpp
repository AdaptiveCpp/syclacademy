/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include "../helpers.hpp"

#include <algorithm>
#include <iostream>

#include <benchmark.h>
#include <image_conv.h>

#include <sycl/sycl.hpp>

inline constexpr util::filter_type filterType = util::filter_type::blur;
inline constexpr int filterWidth = 11;
inline constexpr int halo = filterWidth / 2;

int main() {
  const char* inputImageFile = "Code_Exercises/Images/dogs.png";
  const char* outputImageFile = "Code_Exercises/Images/blurred_dogs.png";

  auto inputImage = util::read_image(inputImageFile, halo);

  auto outputImage = util::allocate_image(
      inputImage.width(), inputImage.height(), inputImage.channels());

  auto filter = util::generate_filter(filterType, filterWidth);

  try {
    sycl::queue myQueue { sycl::gpu_selector_v };

    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    auto inputImgWidth = inputImage.width();
    auto inputImgHeight = inputImage.height();
    auto channels = inputImage.channels();
    auto filterWidth = filter.width();
    auto halo = filter.half_width();

    auto globalRange = sycl::range(inputImgWidth, inputImgHeight);
    auto localRange = sycl::range(1, 32);
    auto ndRange = sycl::nd_range(globalRange, localRange);

    auto inBufRange =
        sycl::range(inputImgHeight + (halo * 2), inputImgWidth + (halo * 2)) *
        sycl::range(1, channels);
    auto outBufRange =
        sycl::range(inputImgHeight, inputImgWidth) * sycl::range(1, channels);

    auto filterRange = filterWidth * sycl::range(1, channels);

    auto inDev = sycl::malloc_device<float>(inBufRange.size(), myQueue);
    auto outDev = sycl::malloc_device<float>(outBufRange.size(), myQueue);
    auto filterDev = sycl::malloc_device<float>(filterRange.size(), myQueue);

    myQueue.copy<float>(inputImage.data(), inDev, inBufRange.size());
    myQueue.copy<float>(filter.data(), filterDev, filterRange.size());

    // synchronize before benchmark, to not measure data transfers.
    myQueue.wait_and_throw();

    util::benchmark(
        [&]() {
          myQueue.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
            auto globalId = item.get_global_id();
            globalId = sycl::id { globalId[1], globalId[0] };

            auto channelsStride = sycl::range(1, channels);
            auto haloOffset = sycl::id(halo, halo);
            auto src = (globalId + haloOffset) * channelsStride;
            auto dest = globalId * channelsStride;

            float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

            for (int r = 0; r < filterWidth; ++r) {
              for (int c = 0; c < filterWidth; ++c) {
                auto srcOffset = sycl::id(src[0] + (r - halo),
                                          src[1] + ((c - halo) * channels));
                auto filterOffset = sycl::id(r, c * channels);

                for (int i = 0; i < 4; ++i) {
                  auto channelOffset = sycl::id(0, i);
                  sum[i] += inDev[srcOffset[0] * inBufRange[1] + srcOffset[1] +
                                  channelOffset[1]] *
                            filterDev[filterOffset[0] * filterRange[1] +
                                      filterOffset[1] + channelOffset[1]];
                }
              }
            }

            for (size_t i = 0; i < 4; ++i) {
              outDev[dest[0] * outBufRange[1] + dest[1] + i] = sum[i];
            }
          });

          myQueue.wait_and_throw();
        },
        100, "image convolution (coalesced)");
    myQueue.copy<float>(outDev, outputImage.data(), outBufRange.size());
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  util::write_image(outputImage, outputImageFile);

  SYCLACADEMY_ASSERT(true);
}
