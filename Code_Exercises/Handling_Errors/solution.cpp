/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include "../helpers.hpp"

#include <sycl/sycl.hpp>


void synchronous_error() {
  auto defaultQueue = sycl::queue { [](const sycl::device& d) { return -1; }};
}

void asynchronous_error() {
  auto asyncHandler = [&](sycl::exception_list exceptionList) {
    for (auto& e : exceptionList) {
      std::rethrow_exception(e);
    }
  };

  auto defaultQueue = sycl::queue { asyncHandler };

  defaultQueue.memset((void*)NULL, 0, 1).wait();

  defaultQueue.wait_and_throw();
}

int main() {
  try {
    synchronous_error();
  } catch (const sycl::exception& e) {
    std::cout << "Synchronous exception caught: " << e.what() << std::endl;
  }

  try {
    asynchronous_error();
  } catch (const sycl::exception& e) {
    std::cout << "Asynchronous exception caught: " << e.what() << std::endl;
  }

  SYCLACADEMY_ASSERT(true);
}
