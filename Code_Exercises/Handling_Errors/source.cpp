/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include "../helpers.hpp"

#include <sycl/sycl.hpp>

int main() {

  // Task: catch synchronous and asynchronous exceptions

  auto defaultQueue = sycl::queue {};

  defaultQueue.parallel_for(sycl::range<1> { 0 }, [=](sycl::id<1> idx) {}).wait();
  defaultQueue.throw_asynchronous();

  SYCLACADEMY_ASSERT(true);
}
