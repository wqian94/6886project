/**
 * Runge-Kutta Library
 *
 * William Qian
 */


#include <math.h>

#include "rk.hh"


#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)


rk_t rk_init(
    uint64_t s, uint64_t dof, void (*f)(double, const double*, double*),
    double fac, double facmax, double facmin) {
  size_t a_len = s * s;
  size_t b_len = s;
  size_t c_len = s;
  double *coeffs = malloc((a_len + b_len + c_len) * (sizeof *coeffs));
  rk_t rk = (rk_t) {
    .s = s,
    .a = coeffs,
    .b = coeffs + a_len * (sizeof *coeffs),
    .c = coeffs + (a_len + b_len) * (sizeof *coeffs),
    .f = f,
    .dof = dof,
    .fac = fac,
    .facmax = facmax,
    .facmin = facmin,
  };
  return rk;
}

void rk_free(rk_t *rk) {
  free(rk->a);
}

// Convert subscripts to indices in a zero-indexed manner
static uint64_t rk_sub2ind(const rk_t *rk, const uint64_t i, const uint64_t j) {
  return i * (i - 1) / 2 + j;
}

uint64_t rk_rk4_eval(
    rk_t *rk, const double *y0, const double x1, const uint64_t nsteps,
    double *y1) {
  const uint8_t dof = rk->dof;
  const double h = x1 / nsteps;
  double x = 0;
  double y1q[dof];

  uint64_t f_evals = 0;

  memcpy(y1q, y0, sizeof y1q);

  for (uint64_t i = 0; i < nsteps; i++) {
    f_evals += rk_rk4_step(rk, x, h, y1q, y1);
    x += h;
    if (i < nsteps - 1) {
      memcpy(y1q, y1, sizeof y1q);  // It would be smarter to do buffer swapping
    }
  }

  return f_evals;
}

uint64_t rk_rk4_step(
    rk_t *rk, const double x, const double h, const double *y0, double *y1) {
  const uint8_t dof = rk->dof;
  const uint64_t s = rk->s;
  double k[s][dof];

  uint64_t f_evals = 0;

  // Zero out y1
  memset(y1, 0, dof * (sizeof *y1));

  // Compute k1 through ks, incrementally updating 
  for (uint64_t i = 0; i < s; i++) {
    double q[dof];
    for (uint8_t d = 0; d < dof; d++) {
      q[d] = y0[d];
      for (uint64_t j = 0; j < i; j++) {
        q[d] += h * rk->a[rk_sub2ind(rk, i, j)] * k[j][d];
      }
    }
    rk->f(x + rk->c[i] * h, q, k[i]);
    f_evals++;

    for (uint8_t d = 0; d < dof; d++) {
      y1[d] += rk->b[i] * k[i][d];
    }
  }

  // Compute final value of y1
  for (uint8_t d = 0; d < dof; d++) {
    y1[d] = y0[d] + h * y1[d];
  }

  return f_evals;
}

uint64_t rk_irk5_step(
    rk_t *rk, const double x, const double h, const double *y0, double *y1,
    const double delta_t) {
  const uint8_t dof = rk->dof;
  const uint64_t s = rk->s;
  double *k = malloc(s * dof * (sizeof *k));
  double *k_prev = malloc(s * dof * (sizeof *k_prev));
  double *q = malloc(dof * (sizeof *q));

  uint64_t f_evals = 0;

  // Zero out y1 and k
  memset(y1, 0, dof * (sizeof *y1));
  memset(k, 0, s * dof * (sizeof k));

  // Iteratively converge on the right values of k
  double deltasq;  // Delta squared
  size_t iterations = 0;
  do {
    if (1000 < iterations++) {
      printf("Too many iterations in IRK5, %s:%d\n", __FILE__, __LINE__);
      abort();
    }

    memmove(k_prev, k, s * dof * (sizeof k));

    // Compute k1 through ks, incrementally updating 
    for (uint64_t i = 0; i < s; i++) {
      for (uint8_t d = 0; d < dof; d++) {
        q[d] = y0[d];
        for (uint64_t j = 0; j < s; j++) {
          q[d] += h * rk->a[rk_sub2ind(rk, i, j)] * k_prev[j  * dof + d];
        }
      }
      rk->f(rk, x + rk->c[i] * h, q, k + i * dof);
      f_evals++;
    }

    // Compute error
    deltasq = 0;
    for (uint64_t i = 0; i < s; i++) {
      for (uint8_t d = 0; d < dof; d++) {
        deltasq +=
          (k[i * dof + d] - k_prev[i * dof + d]) *
          (k[i * dof + d] - k_prev[i * dof + d]);
      }
    }

  } while (deltasq > delta_t * delta_t);

  // Compute y1
  for (uint8_t d = 0; d < dof; d++) {
    for (uint64_t i = 0; i < s; i++) {
      y1[d] += rk->b[i] * k[i * dof + d];
    }
    y1[d] = y0[d] + h * y1[d];
  }
  free(k);
  free(k_prev);
  free(q);

  return f_evals;
}

uint64_t rk_irk5_eval(
    rk_t *rk, const double *y0, const double x1, const uint64_t nsteps,
    double *y1, const double delta_t) {
  const uint8_t dof = rk->dof;
  const double h = x1 / nsteps;
  double x = 0;
  double y1q[dof];

  uint64_t f_evals = 0;

  memcpy(y1q, y0, sizeof y1q);

  for (uint64_t i = 0; i < nsteps; i++) {
    f_evals += rk_irk5_step(rk, x, h, y1q, y1, delta_t);
    x += h;
    if (i < nsteps - 1) {
      memcpy(y1q, y1, sizeof y1q);  // It would be smarter to do buffer swapping
    }
  }

  return f_evals;
}

uint64_t rk_irk5_eval_collect(
    rk_t *rk, const double *y0, const double x1, const uint64_t nsteps,
    double *y1, const double delta_t) {
  const uint8_t dof = rk->dof;
  const double h = x1 / nsteps;
  double x = 0;

  uint64_t f_evals = 0;

  memcpy(y1, y0, dof * (sizeof *y1));

  for (uint64_t i = 0; i < nsteps; i++) {
    f_evals += rk_irk5_step(
      rk, x, h, y1 + dof * i, y1 + dof * (i + 1), delta_t);

    x += h;
  }

  return f_evals;
}

// Returns the Hermite interpolation at the given values
double rk_fsal_dense_hermite(
    const double x, const double h, const double t,
    const double y0, const double y1, const double dy0, const double dy1) {
  return (1 - t) * y0 + t * y1 + t * (t - 1) * ((1 - 2 * t) * (y1 - y0)
    + (t - 1) * h * dy0 + t * h * dy1);
}

// Populates with dense output, returns the next index
uint64_t rk_fsal_dense(
    rk_t *rk, const double x, const double h, const double interval,
    uint64_t index, const double *y0, const double *y1, const double *dy0,
    const double *dy1, double *dense) {
  while (interval * index < x + h) {
    double t = (interval * index - x) / h;
    for (uint8_t d = 0; d < rk->dof; d++) {
      dense[index * rk->dof + d] =
        rk_fsal_dense_hermite(x, h, t, y0[d], y1[d], dy0[d], dy1[d]);
    }
    index++;
  }

  return index;
}

uint64_t rk_fsal_eval(
    rk_t *rk, const double *bh, const double *y0, const double x1,
    const double atol, const double rtol, double *y1, series_t *series,
    const double di, double **dense) {
  const uint8_t dof = rk->dof;
  uint64_t f_evals = 0;
  double x = 0;
  double h = 0.01;
  double y1q[dof];  // Temporary buffer for y1 being used as y0 in the next step
  double y1h[dof];
  double k[dof];  // Carryover k
  double knext[dof];

  const uint64_t nintervals = (uint64_t)(x1 / di) + 1;  // Max # intervals
  if (dense) {
    *dense = malloc(dof * nintervals * (sizeof **dense));
  }
  size_t dindex = 0;  // Dense output index

  size_t ssize = 8192;
  if (series) {
    series->dof = dof + 1;
    series->size = 0;
    series->data = malloc(ssize);
  }

  // Set up for first iteration
  memcpy(y1q, y0, sizeof y1q);
  rk->f(x + rk->c[0] * h, y0, k);
  f_evals++;

  while (x < x1) {
    f_evals += rk_fsal_step(rk, k, knext, x, h, bh, y1q, y1, y1h);
    double h0 = h;
    if (rk_fsal_nextstep(rk, atol, rtol, h, &h, y1, y1h)) {
      if (series) {
        size_t snext = (series->size + 1) * series->dof - 1;
        if (snext * (sizeof *series->data) >= ssize) {
          series->data = realloc(series->data, ssize *= 2);
        }
        series->data[series->size * series->dof] = x;
        for (uint8_t d = 0; d < dof; d++) {
          series->data[series->size * series->dof + d + 1] = y1[d];
        }
        series->size++;
      }
      if (dense) {
        dindex = rk_fsal_dense(rk, x, h, di, dindex, y1q, y1, k, knext, *dense);
      }
      if ((x += h0) >= x1) {
        break;
      }
      memcpy(y1q, y1, sizeof y1q);  // It would be smarter to do buffer swapping
      memcpy(k, knext, sizeof k);
    }
    if (x + h > x1) {
      h = x1 - x;
    }
  }

  if (series) {
    series->data = realloc(
      series->data, series->size * series->dof * (sizeof *series->data));
  }

  return f_evals;
}

uint64_t rk_fsal_step(
    rk_t *rk, const double *k1, double *ks, const double x, const double h,
    const double *bh, const double *y0, double *y1, double *y1h) {
  const uint8_t dof = rk->dof;
  const uint64_t s = rk->s;
  double k[s][dof];

  uint64_t f_evals = 0;

  // Zero out y1 and y1h
  memset(y1, 0, dof * (sizeof *y1));
  memset(y1h, 0, dof * (sizeof *y1h));

  // Compute k1 through ks, incrementally updating 
  for (uint64_t i = 0; i < s; i++) {
    if (!i && k1) {
      memcpy(k[i], k1, sizeof k[i]);
    } else {
      double q[dof];
      for (uint8_t d = 0; d < dof; d++) {
        q[d] = y0[d];
        for (uint64_t j = 0; j < i; j++) {
          q[d] += h * rk->a[rk_sub2ind(rk, i, j)] * k[j][d];
        }
      }
      rk->f(x + rk->c[i] * h, q, k[i]);
      f_evals++;
    }

    for (uint8_t d = 0; d < dof; d++) {
      y1[d] += rk->b[i] * k[i][d];
      y1h[d] += bh[i] * k[i][d];
    }
  }

  // Compute final values of y1 and y1h
  for (uint8_t d = 0; d < dof; d++) {
    y1[d] = y0[d] + h * y1[d];
    y1h[d] = y0[d] + h * y1h[d];
  }

  // Copy-cache ks
  memcpy(ks, k[s - 1], sizeof k[s - 1]);

  return f_evals;
}

bool rk_fsal_nextstep(
    rk_t *rk, const double atol, const double rtol, double h, double *hnew,
    const double *y1, const double *y1h) {
  // Compute error first
  double sum = 0;
  for (uint64_t i = 0; i < rk->dof; i++) {
    double sc = atol + rtol * max(fabs(y1[i]), fabs(y1h[i]));
    sum += ((y1[i] - y1h[i]) * (y1[i] - y1h[i])) / (sc * sc);

  }
  double err = sqrt(sum / rk->dof);

  // Compute hnew
  uint64_t q = rk->s - 2;
  *hnew = h * min(rk->facmax, max(rk->facmin,
                                  rk->fac * pow(1 / err, 1. / (q + 1))));

  return err < 1;
}

void series_free(series_t *series) {
  free(series->data);
}
