/**
 * Benchmark driver for 6.886 project
 *
 * William Qian
 */

#include <stdio.h>
#include <thread>
#include <vector>

#include "project.hh"

void test_brusselnator() {
  // Brusselnator: f(x, y0, y1) = (1 + y0^2y1 - 4y0, 3y0 - y0^2y1)
  auto brusselnator_f = [] (const double x, const double in[2], double out[2]) {
    out[0] = 1 + in[0] * (in[0] * in[1] - 4);
    out[1] = in[0] * (3 - in[0] * in[1]); 
  };

  const size_t nthreads = 3;
  using rk_type = rk::RK4Classic<3>;
  using wl_type = bench::Workload<rk_type, nthreads>;
  using gen_type = wl_type::Generator;

  auto rk4 = rk_type();
  auto workload = wl_type::PeanoGrid(0, {0, 0, 0}, {1, 1, 1}, 2);

  auto thread_runner = [brusselnator_f] (
      rk_type& rk4, wl_type& workload, double& seconds, size_t threadid) {
    auto gen = workload.generator(threadid);

    auto driver = [brusselnator_f] (rk_type& rk4, gen_type& gen) {
      for (auto state = gen.next(); state; state = gen.next()) {
        rk4.explicit_eval(
          brusselnator_f, (*state)->x, (*state)->y, (*state)->x + 1, 300000);
      }
    };

    seconds = bench::Util::timeit(driver, rk4, gen);
  };

  std::vector<std::thread> threads;
  std::vector<double> seconds(nthreads, 0);
  for (size_t th = 0; th < nthreads; th++) {
    threads.emplace_back(
        thread_runner, std::ref(rk4), std::ref(workload),
        std::ref(seconds[th]), th);
  }

  for (auto& th : threads) {
    th.join();
  }

  double work = 0;
  double span = 0;
  for (const double secs : seconds) {
    printf("%f\n", secs);
    work += secs;
    span = std::max(span, secs);
  }

  /*
  for (auto itr = states.begin(); itr != states.end(); itr++) {
    printf("(%.6f, %.6f, %.6f)\n", itr->x, itr->y[0], itr->y[1]);
  }
  */

  printf(
      "Threads: %zu, Work: %.6f seconds, Span: %.6f seconds\n",
      threads.size(), work, span);
}

void test_brusselnator2() {
  // Brusselnator: f(x, y0, y1) = (1 + y0^2y1 - 4y0, 3y0 - y0^2y1)
  auto brusselnator_f = [] (const double x, const double in[2], double out[2]) {
    out[0] = 1 + in[0] * (in[0] * in[1] - 4);
    out[1] = in[0] * (3 - in[0] * in[1]); 
  };

  const size_t nthreads = 3;
  using rk_type = rk::IRK5Geng<3>;
  using wl_type = bench::Workload<rk_type, nthreads>;
  using gen_type = wl_type::Generator;

  auto irk5 = rk_type();
  auto workload = wl_type::PeanoGrid(0, {0, 0, 0}, {1, 1, 1}, 2);

  auto thread_runner = [brusselnator_f] (
      rk_type& irk5, wl_type& workload, double& seconds, size_t threadid) {
    auto gen = workload.generator(threadid);

    auto driver = [brusselnator_f] (rk_type& irk5, gen_type& gen) {
      for (auto state = gen.next(); state; state = gen.next()) {
        if ((*state)->aborted) {  // Skip previously-aborted points
          continue;
        }

        irk5.implicit_eval(
          brusselnator_f, (*state)->x, (*state)->y, (*state)->x + 1, 300000,
          1e-12);
      }
    };

    seconds = bench::Util::timeit(driver, irk5, gen);
  };

  std::vector<std::thread> threads;
  std::vector<double> seconds(nthreads, 0);
  for (size_t th = 0; th < nthreads; th++) {
    threads.emplace_back(
        thread_runner, std::ref(irk5), std::ref(workload),
        std::ref(seconds[th]), th);
  }

  for (auto& th : threads) {
    th.join();
  }

  double work = 0;
  double span = 0;
  for (const double secs : seconds) {
    work += secs;
    span = std::max(span, secs);
  }

  /*
  for (auto itr = states.begin(); itr != states.end(); itr++) {
    printf("(%.6f, %.6f, %.6f)\n", itr->x, itr->y[0], itr->y[1]);
  }
  */

  printf(
      "Threads: %zu, Work: %.6f seconds, Span: %.6f seconds\n",
      threads.size(), work, span);

  printf("Aborted points: %zu\n", workload.stat_aborted());
}

int main(int argc, char* arg[]) {
  test_brusselnator();
  test_brusselnator2();
  return 0;
}
