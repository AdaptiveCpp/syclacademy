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

inline constexpr util::filter_type filterType = util::filter_type::identity;
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
    sycl::queue myQueue {
      // sycl::gpu_selector_v
    };

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
    auto inputImageSize = inputImage.size();
    auto outputImageSize = outputImage.size();
    auto filterSize = filter.size();

    auto inBufRange =
        sycl::range(inputImgHeight + (halo * 2), inputImgWidth + (halo * 2)) *
        sycl::range(1, channels);
    auto outBufRange =
        sycl::range(inputImgHeight, inputImgWidth) * sycl::range(1, channels);

    auto filterRange = filterWidth * sycl::range(1, channels);

    float* inputImageDev =
        sycl::malloc_device<float>(inBufRange[0] * inBufRange[1], myQueue);
    float* outputImagDev = sycl::malloc_device<float>(outputImageSize, myQueue);
    float* filterDev = sycl::malloc_device<float>(filterSize, myQueue);

    myQueue.copy<float>(inputImage.data(), inputImageDev,
                        inBufRange[0] * inBufRange[1]);
    myQueue.copy<float>(filter.data(), filterDev, filterSize);
    // synchronize before benchmark, to not measure copy
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

                for (int i = 0; i < 4; ++i) {
                  sum[i] += inputImageDev[srcOffset[0] * inBufRange[1] +
                                          srcOffset[1] + i] *
                            filterDev[r * filterWidth + c * channels + i];
                }
              }
            }

            for (size_t i = 0; i < 4; ++i) {
              outputImagDev[dest[0] * outBufRange[1] + dest[1] + i] = sum[i];
            }
          });

          myQueue.wait_and_throw();
        },
        100, "image convolution (coalesced)");
    myQueue.copy<float>(outputImagDev, outputImage.data(), outputImageSize)
        .wait_and_throw();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  util::write_image(outputImage, outputImageFile);

  SYCLACADEMY_ASSERT(true);
}
