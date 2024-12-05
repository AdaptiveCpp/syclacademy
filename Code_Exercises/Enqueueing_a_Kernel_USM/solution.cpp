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
  auto defaultQueue = sycl::queue {};

  defaultQueue.single_task([=]() { sycl::detail::print("Hello World!\n"); }).wait();

  SYCLACADEMY_ASSERT(true);
}
