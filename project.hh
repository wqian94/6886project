/*
 * Benchmark library for 6.886 project
 *
 * William Qian
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <thread>
#include <vector>

#include "rk.hh"

namespace bench {

class Util {
public:
  template <typename Function, typename... Args>
  static double timeit(const Function& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
  }

  static bool pin_thread(std::thread& thread, size_t threadid) {
    const auto num_cores = std::thread::hardware_concurrency();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(threadid % num_cores, &cpuset);
    auto success = !pthread_setaffinity_np(
        thread.native_handle(), sizeof cpuset, &cpuset);
    return success;
  }

  static void reset() {
    started_.store(false);
  }

  static void start() {
    started_.store(true);
  }

  static void wait_for_start() {
    while (!started_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  static std::atomic<bool> started_;
};

template <typename T, size_t THREADS=1>
class Workload {
public:
  typedef typename T::State State;
  typedef std::initializer_list<double> Tuple;
  typedef std::vector<double> Vector;

  class Generator{
  public:
    typedef typename Workload::iterator_type iterator_type;
    typedef std::optional<State*> value_type;

    Generator(iterator_type begin_itr, iterator_type end_itr) :
        begin_itr_(begin_itr), end_itr_(end_itr), itr_(begin_itr),
        reversed(false) {}

    // Produce a pointer to the next State, if there is one
    value_type next() {
      if (reversed) {
        if (itr_ <= begin_itr_) {  // Finished reversing
          return value_type();
        } else {
          return &*(--itr_);
        }
      } else {
        if (itr_ >= end_itr_) {  // Finished forward stepping
          return value_type();
        } else {
          return &*(itr_++);
        }
      }
    }

    // Reverse the direction of iteration
    void reverse() {
      reversed = !reversed;
    }

  private:
    iterator_type begin_itr_;
    iterator_type end_itr_;
    iterator_type itr_;
    bool reversed;
  };

  Workload() {}

  // Workload based on a Peano curve starting at y0 (inclusive) with the given
  // side lengths, using the given iteration depth -- resulting in 3^depth
  // number of points per dimension
  static Workload PeanoGrid(double x, Tuple y0, Tuple lengths, size_t depth) {
    Workload w;
    w.peano_helper(
        x, Vector(y0.begin(), y0.end()), Vector(lengths.begin(), lengths.end()),
        0, 0, depth);
    return w;
  }

  // Workload based on a standard grid starting at y0 (inclusive) with the given
  // side lengths, using the given iteration depth -- resulting in 3^depth
  // number of points per dimension
  static Workload StandardGrid(double x, Tuple y0, Tuple lengths, size_t depth) {
    Workload w;
    w.standard_helper(
        x, Vector(y0.begin(), y0.end()), Vector(lengths.begin(), lengths.end()),
        0, 0, depth);
    return w;
  }

  void add_unit(double x, Tuple y) {
    values_.emplace_back(State(x, y));
  }

  void add_unit(double x, Vector y) {
    values_.emplace_back(State(x, y));
  }

  // Begin (inclusive) and end (exclusive) iterators for this thread; assumes
  // that no new points are added after this is called
  //
  // threadid is in the range [0, THREADS)
  Generator generator(const size_t threadid=0) {
    const size_t block_offset = values_.size() / THREADS;
    if (threadid + 1 == THREADS) {  // Last thread
      return Generator(
          values_.begin() + threadid * block_offset, values_.end());
    }

    return Generator(
        values_.begin() + threadid * block_offset,
        values_.begin() + (threadid + 1) * block_offset);
  }

  size_t stat_aborted() const {
    size_t count = 0;
    for (auto itr = values_.begin(); itr != values_.end(); itr++) {
      count += !!itr->aborted;
    }
    return count;
  }

private:
  typedef std::vector<State> container_type;
  typedef typename container_type::iterator iterator_type;
  static const uint64_t ORIENTATION_MASK = 0x5555555555555555ULL;

  void peano_helper(
      double x, Vector y0, Vector lengths, size_t dimension, size_t depth,
      size_t max_depth, uint64_t orientation=0) {
    if (dimension >= T::DIMENSIONS) {
      return;
    }

    const auto lowest_depth = (depth + 1 >= max_depth);
    const auto last_dimension = (dimension + 1 >= T::DIMENSIONS);
    const bool forward = !(orientation & (1ULL << dimension));
    const uint64_t orientation_inv = (orientation ^ ~(1ULL << dimension));

    Vector next_lengths(lengths.begin(), lengths.end());
    next_lengths[dimension] /= 3;

    Vector y1(y0);
    Vector y2(y0);
    Vector y3(y0);

    if (forward) {
      y2[dimension] = y1[dimension] + next_lengths[dimension];
      y3[dimension] = y2[dimension] + next_lengths[dimension];
    } else {
      y2[dimension] = y3[dimension] + next_lengths[dimension];
      y1[dimension] = y2[dimension] + next_lengths[dimension];
    }

    /*
    printf(
        "DIM: %zu, DEP: %zu, ORI: %lx, FOR: %d, y0: %f, (y1, y2, y3): %f %f %f\n",
        dimension, depth, orientation, forward, y0[dimension], y1[dimension],
        y2[dimension], y3[dimension]);
    */

    if (lowest_depth && last_dimension) {
      add_unit(x, y1);
      add_unit(x, y2);
      add_unit(x, y3);
    } else {  // Split dimension
      if (last_dimension) {
        peano_helper(
            x, y1, next_lengths, 0, depth + 1, max_depth, orientation);
        peano_helper(
            x, y2, next_lengths, 0, depth + 1, max_depth, orientation_inv);
        peano_helper(
            x, y3, next_lengths, 0, depth + 1, max_depth, orientation);
      } else {
        peano_helper(
            x, y1, next_lengths, dimension + 1, depth, max_depth, orientation);
        peano_helper(
            x, y2, next_lengths, dimension + 1, depth, max_depth,
            orientation_inv);
        peano_helper(
            x, y3, next_lengths, dimension + 1, depth, max_depth, orientation);
      }
    }
  }

  void standard_helper(
      double x, Vector y0, Vector lengths, size_t dimension, size_t depth,
      size_t max_depth, uint64_t orientation=0) {
    if (dimension >= T::DIMENSIONS) {
      return;
    }

    const auto lowest_depth = (depth + 1 >= max_depth);
    const auto last_dimension = (dimension + 1 >= T::DIMENSIONS);

    Vector next_lengths(lengths.begin(), lengths.end());
    next_lengths[dimension] /= 3;

    Vector y1(y0);
    Vector y2(y0);
    Vector y3(y0);

    y2[dimension] = y1[dimension] + next_lengths[dimension];
    y3[dimension] = y2[dimension] + next_lengths[dimension];

    if (lowest_depth && last_dimension) {
      add_unit(x, y1);
      add_unit(x, y2);
      add_unit(x, y3);
    } else {  // Recursively split
      if (last_dimension) {
        standard_helper(x, y1, next_lengths, 0, depth + 1, max_depth);
        standard_helper(x, y2, next_lengths, 0, depth + 1, max_depth);
        standard_helper(x, y3, next_lengths, 0, depth + 1, max_depth);
      } else {
        standard_helper(x, y1, next_lengths, dimension + 1, depth, max_depth);
        standard_helper(x, y2, next_lengths, dimension + 1, depth, max_depth);
        standard_helper(x, y3, next_lengths, dimension + 1, depth, max_depth);
      }
    }
  }

  container_type values_;
};

}  // namespace bench
