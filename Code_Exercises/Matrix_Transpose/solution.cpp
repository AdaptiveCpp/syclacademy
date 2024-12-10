/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 The tiling should work as follows:

 The groupOffset will need to be inverted, as well as the localId.

 In:                                      Out:
 +----------------------------+           +----------------------------+
 |             group  ^       |           |       ^                    |
 |           Offset[1]|       |           |       |                    |
 |                    V       |           |       |                    |
 |<------------------>+-------+           |       |                    |
 |   groupOffset[0]   |1     2|           |       |groupOffset[0]      |
 |                    | tile  |           |       |                    |
 |                    |3     4| ------->  |       |                    |
 |                    +-------+           |       V                    |
 |                            |           |       +-------+            |
 |                            |           |       |1     3|            |
 |                            |           |       | tile  |            |
 |                            |           |       |2     4|            |
 +----------------------------+           +-------+-------+------------+
                                           <----->
                                           groupOffset[1]

Within a tile, each work item is assigned to a single value:

InTile:               OutTile:
+------------+        +------------+
|   local  ^ |        |  local^    |
|   Id[1]  | |        |  Id[0]|    |
|          V |------->|       |    |
|<-------->* |        |       V    |
|localId[0]  |        |<----->*    |
+------------+        +------------+
                      localId[1]

*/

#include "../helpers.hpp"

#include <hipSYCL/sycl/libkernel/item.hpp>
#include <hipSYCL/sycl/usm.hpp>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

#include <benchmark.h>

constexpr size_t N = 8192;
constexpr size_t numIters = 100;

using T = float;

int main() {

  std::vector<T> A(N * N);
  std::vector<T> A_T(N * N);
  std::vector<T> A_T_comparison(N * N);

  for (auto i = 0; i < N * N; ++i) {
    A[i] = i;
  }

  try {
    auto q = sycl::queue {};

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    sycl::range globalRange { N, N };
    sycl::range localRange { 16, 16 };
    sycl::nd_range ndRange { globalRange, localRange };

    auto in_D = sycl::malloc_device<T>(A.size(), q);
    auto out_D = sycl::malloc_device<T>(A_T.size(), q);
    auto comp_D = sycl::malloc_device<T>(A_T_comparison.size(), q);

    q.copy<T>(A.data(), in_D, A.size()).wait();

    util::benchmark(
        [&]() {
          q.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
            auto id = item.get_global_id();
            auto linearId = id[0] * item.get_global_range(1) + id[1];
            auto transposedId = id[1] * item.get_global_range(0) + id[0];
            comp_D[transposedId] = in_D[linearId];
          });
          q.wait_and_throw();
        },
        numIters, "Naive matrix transpose");

    util::benchmark(
        [&]() {
          q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<T, 2> localAcc { localRange, cgh };

            cgh.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
              // This kernel assumes that localRange[0] == localRange[1]
              auto globalId = item.get_global_id();
              auto localId = item.get_local_id();
              auto groupOffset = globalId - localId;
              auto groupOffset_T = sycl::id { groupOffset[1], groupOffset[0] };

              // Read from global memory in row major and write to local
              // memory in column major
              localAcc[localId[1]][localId[0]] =
                  in_D[globalId[0] * item.get_global_range(0) + globalId[1]];

              // We need to wait here to ensure that all work items have
              // written to local memory before we start reading from it.
              sycl::group_barrier(item.get_group());

              // Read from local memory in row major and write to global
              // memory in row major fashion
              auto storeId = groupOffset_T + localId;
              out_D[storeId[0] * item.get_global_range(0) + storeId[1]] =
                  localAcc[localId[0]][localId[1]];
            });
          });
          q.wait_and_throw();
        },
        numIters, "Tiled local memory matrix transpose");

    q.copy<T>(out_D, A_T.data(), A.size());
    q.copy<T>(comp_D, A_T_comparison.data(), A.size());
    q.wait_and_throw();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (auto i = 0; i < N * N; ++i) {
    SYCLACADEMY_ASSERT(A_T[i] == A_T_comparison[i]);
  }
}
