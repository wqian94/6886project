/**
 * Runge-Kutta IRK5 implementation
 *
 * William Qian
 */

#pragma once

#include <math.h>

#include "rk-base.hh"

namespace rk {

template <size_t DIM>
class IRK5Geng : public RungeKutta<3, DIM> {
public:
  using base=RungeKutta<3, DIM>;
  using typename base::State;
  using typename base::func_type;

  IRK5Geng() : RungeKutta<3, DIM>() {
    a[0][0] = (16 - sqrt(6)) / 72;
    a[0][1] = (328 - 167 * sqrt(6)) / 1800;
    a[0][2] = (-2 + 3 * sqrt(6)) / 450;
    a[1][0] = (326 + 167 * sqrt(6)) / 1800;
    a[1][1] = (16 + sqrt(6)) / 72;
    a[1][2] = (-2 - 3 * sqrt(6)) / 450;
    a[2][0] = (85 - 10 * sqrt(6)) / 180;
    a[2][1] = (85 + 10 * sqrt(6)) / 180;
    a[2][2] = 1. / 18;
    b[0] = (16 - sqrt(6)) / 36;
    b[1] = (16 + sqrt(6)) / 36;
    b[2] = 1. / 9;
    c[0] = (4 - sqrt(6)) / 10;
    c[1] = (4 + sqrt(6)) / 10;
    c[2] = 1;
  }

  State explicit_eval(
      func_type, const double, const double[], const double, const uint64_t) {
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
