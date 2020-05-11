/**
 * Runge-Kutta Base Library
 *
 * William Qian
 */

#pragma once

#include <functional>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>

#define FSAL 0

namespace rk {

// Base structure for a Runge-Kutta scheme with DIM dimensions and STAGES stages
template <size_t STAGES, size_t DIM>
class RungeKutta {
public:
  static constexpr size_t DIMENSIONS = DIM;
  typedef std::function<void(double, const double[DIM], double[DIM])> func_type;

  struct State {
    State(double x) :
        aborted(false), running(true), solved(false), unsupported(false),
        evals(0), x(x) {}
    State(double x, std::initializer_list<double> yinit):
        aborted(false), running(true), solved(false), unsupported(false),
        evals(0), x(x) {
      setY(yinit);
    }
    State(double x, std::vector<double> yinit):
        aborted(false), running(true), solved(false), unsupported(false),
        evals(0), x(x) {
      setY(yinit);
    }

    void setY(std::initializer_list<double> yinit) {
      auto itr = yinit.begin();
      for (size_t i = 0; i < DIM; i++) {
        y[i] = *itr++;
      }
    }

    void setY(std::vector<double> yinit) {
      auto itr = yinit.begin();
      for (size_t i = 0; i < DIM; i++) {
        y[i] = *itr++;
      }
    }

    bool aborted;  // Whether the computation was aborted
    bool running; // Whether the computing is still running
    bool solved;  // Whether the problem was solved
    bool unsupported;  // Whether the selected method is unsupported
    uint64_t evals;  // Number of target function evaluations
    double x;  // Independent dimension
    double y[DIM];  // Dependent dimensions
  };

#if FSAL
  RungeKutta() : fac(0), facmax(0), facmin(0) {}
  RungeKutta(double fac, double facmax, double facmin) :
    fac(fac), facmax(facmax), facmin(facmin) {}
#else
  RungeKutta() {}
#endif

  // Explicit RK
  //
  // diffeq is the differential equation in the initial value problem
  // x0, y0 are the initial values in the initial value problem
  // x1 is the target independent dimension value
  // steps is the number of iterations to go through
  //
  // Returns the final computed State
  State explicit_eval(
      func_type diffeq, const double x0, const double y0[DIM], const double x1,
      const uint64_t steps) const;

  // Implicit RK
  //
  // diffeq is the differential equation in the initial value problem
  // x0, y0 are the initial values in the initial value problem
  // x1 is the target independent dimension value
  // steps is the number of iterations to go through
  // delta_t is the linear error threshold
  // max_iterations is the max number of iterations to try, defaults to 1000
  //
  // Returns the final computed State
  State implicit_eval(
      func_type diffeq, const double x0, const double y0[DIM], const double x1,
      const uint64_t steps, const double delta_t,
      const size_t max_iterations=1000) const;

protected:
  // Converts from matrix index to array index
  size_t to_index(const size_t row, const size_t column) const {
    return row * STAGES + column;
  }

  // Explicit RK step
  //
  // state is the present state
  // diffeq is the differential equation in the initial value problem
  // h is the step size
  // y0 is the current dependent dimensions
  // y1 is the buffer for the next iteration of dependent dimensions
  void explicit_step(
      State& state, func_type diffeq, const double h, const double y0[DIM],
      double y1[DIM]) const;

  // Implicit RK step
  //
  // state is the present state
  // diffeq is the differential equation in the initial value problem
  // h is the step size
  // y0 is the current dependent dimensions
  // delta_t is the linear error threshold
  // max_iterations is the number of iterations to try
  // y1 is the buffer for the next iteration of dependent dimensions
  void implicit_step(
      State& state, func_type diffeq, const double h, const double y0[DIM],
      const double delta_t, const size_t max_iterations, double y1[DIM]) const;

#if FSAL
  double fac;  // Safety factor
  double facmax;
  double facmin;
#endif

  double a[STAGES][STAGES];  // Coefficients
  double b[STAGES];  // Coefficients
  double c[STAGES];  // Coefficients
};

template <size_t STAGES, size_t DIM>
typename RungeKutta<STAGES, DIM>::State RungeKutta<STAGES, DIM>::explicit_eval(
    func_type diffeq, const double x0, const double y0[DIM], const double x1,
    const uint64_t steps) const {
  const double h = (x1 - x0) / steps;  // Step size

  State state(x0);
  double ybuf[DIM];
  for (uint64_t step = 0; step < steps; step++) {
    if (step == 0) {
      explicit_step(state, diffeq, h, y0, state.y);
    } else if (step % 2) {
      explicit_step(state, diffeq, h, state.y, ybuf);
    } else {
      explicit_step(state, diffeq, h, ybuf, state.y);
    }

    if (state.aborted) {
      break;
    }
  }

  if (steps && (steps % 2 == 0)) {  // Even # of steps > 0 means ybuf is final
    memcpy(state.y, ybuf, sizeof ybuf);
  }

  state.x = x1;
  state.solved = !state.aborted;
  state.running = false;

  return state;
}

template <size_t STAGES, size_t DIM>
void RungeKutta<STAGES, DIM>::explicit_step(
    State& state, func_type diffeq, const double h, const double y0[DIM],
    double y1[DIM]) const {
  double k[STAGES][DIM];  // Doesn't need to be zeroed because it's not used
                          // before it's assigned

  // Compute intermediate k's
  for (uint64_t stage = 0; stage < STAGES; stage++) {
    double q[DIM];
    for (size_t d = 0; d < DIM; d++) {
      q[d] = y0[d];
      // Assume lower triangular because explicit RK
      for (uint64_t substage = 0; substage < stage; substage++) {
        q[d] += h * a[stage][substage] * k[substage][d];
      }
    }

    // Invoke the diffeq computation
    diffeq(state.x + c[stage] * h, q, k[stage]);
    state.evals++;

    for (size_t d = 0; d < DIM; d++) {
      if (stage) {
        y1[d] += b[stage] * k[stage][d];
      } else {  // First stage
        y1[d] = b[stage] * k[stage][d];
      }
    }
  }

  // Compute final value of y1
  for (size_t d = 0; d < DIM; d++) {
    y1[d] = y0[d] + h * y1[d];
  }
}

template <size_t STAGES, size_t DIM>
typename RungeKutta<STAGES, DIM>::State RungeKutta<STAGES, DIM>::implicit_eval(
    func_type diffeq, const double x0, const double y0[DIM], const double x1,
    const uint64_t steps, const double delta_t, const size_t max_iterations) const {
  const double h = (x1 - x0) / steps;  // Step size

  State state(x0);
  double ybuf[DIM];
  for (uint64_t step = 0; step < steps; step++) {
    if (step == 0) {
      implicit_step(state, diffeq, h, y0, delta_t, max_iterations, state.y);
    } else if (step % 2) {
      implicit_step(state, diffeq, h, state.y, delta_t, max_iterations, ybuf);
    } else {
      implicit_step(state, diffeq, h, ybuf, delta_t, max_iterations, state.y);
    }

    if (state.aborted) {
      //int d;
      //scanf("%d\n", &d);
      break;
    }
  }

  if (steps && (steps % 2 == 0)) {  // Even # of steps > 0 means ybuf is final
    memcpy(state.y, ybuf, sizeof ybuf);
  }

  state.x = x1;
  state.solved = !state.aborted;
  state.running = false;

  return state;
}

template <size_t STAGES, size_t DIM>
void RungeKutta<STAGES, DIM>::implicit_step(
    State& state, func_type diffeq, const double h, const double y0[DIM],
    const double delta_t, const size_t max_iterations, double y1[DIM]) const {
  double k[2][STAGES][DIM];
  memset(k, 0, sizeof k);

  double delta_sq = 0;
  size_t iteration = 0;

  do {
    if (iteration++ > max_iterations) {
      state.aborted = true;
      return;
    }

    // Compute intermediate k's
    for (uint64_t stage = 0; stage < STAGES; stage++) {
      double q[DIM];
      for (size_t d = 0; d < DIM; d++) {
        q[d] = y0[d];
        // Assume lower triangular because explicit RK
        for (uint64_t substage = 0; substage < stage; substage++) {
          q[d] += h * a[stage][substage] * k[iteration % 2][substage][d];
        }
      }

      // Invoke the diffeq computation
      diffeq(state.x + c[stage] * h, q, k[(iteration + 1) % 2][stage]);
      state.evals++;
    }

    // Compute error
    delta_sq = 0;
    for (size_t stage = 0; stage < STAGES; stage++) {
      for (size_t d = 0; d < DIM; d++) {
        delta_sq +=  // Squares don't care about signedness :)
          (k[1][stage][d] - k[0][stage][d]) * (k[1][stage][d] - k[0][stage][d]);
      }
    }
    //printf("%f\n", delta_sq);
  } while (delta_sq > delta_t * delta_t);

  // Compute final value of y1
  for (size_t d = 0; d < DIM; d++) {
    for (size_t stage = 0; stage < STAGES; stage++) {
      if (stage) {
        y1[d] += b[stage] * k[iteration % 2][stage][d];
      } else {  // First stage
        y1[d] = b[stage] * k[iteration % 2][stage][d];
      }
    }
    y1[d] = y0[d] + h * y1[d];
  }
}

}  // namespace rk
