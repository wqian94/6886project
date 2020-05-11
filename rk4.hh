/**
 * Runge-Kutta RK4 implementation
 *
 * William Qian
 */

#pragma once

#include "rk-base.hh"

namespace rk {

template <size_t DIM>
class RK4Classic : public RungeKutta<4, DIM> {
public:
  using base=RungeKutta<4, DIM>;
  using typename base::State;
  using typename base::func_type;

  RK4Classic() : RungeKutta<4, DIM>() {
    a[1][0] = a[2][1] = 0.5;
    a[3][2] = 1;
    a[0][0] = a[0][1] = a[0][2] = a[0][3] = 0;
    a[1][1] = a[1][2] = a[1][3] = 0;
    a[2][0] = a[2][2] = a[2][3] = 0;
    a[3][0] = a[3][1] = a[3][3] = 0;
    b[0] = b[3] = 1. / 6;
    b[1] = b[2] = 1. / 3;
    c[0] = 0;
    c[1] = c[2] = 0.5;
    c[3] = 1;
  }

  State implicit_eval(
      func_type, const double, const double[], const double, const uint64_t,
      const double, const size_t) {
    State state(0);
    state.aborted = state.unsupported = true;
    state.running = false;
    return state;
  }

protected:
  using base::a;
  using base::b;
  using base::c;
};

}  // namespace rk
