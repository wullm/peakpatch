/*******************************************************************************
 * This file is part of Peak.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>

#include "../include/peak.h"

typedef int (*func_ptr)(void *params, ...);

struct ellipt_params {
    double delta0;
    double c[3];
    double *D_vec;
    double *a_vec;
    double *addot_vec;
    double *OmH2_vec;
    double f_collapse;
    struct strooklat *spline;
};

int f(double t, const double y[], double f[], void *params) {
    struct ellipt_params *ep = (struct ellipt_params *) params;

    double OmH2 = strooklat_interp(ep->spline, ep->OmH2_vec, t);
    double D = strooklat_interp(ep->spline, ep->D_vec, t);
    double a = strooklat_interp(ep->spline, ep->a_vec, t);
    double addot = strooklat_interp(ep->spline, ep->addot_vec, t);
    double *c = ep->c;

    // OmH2 = 0.00145;

    double delta = a*a*a / (y[0] * y[1] * y[2]) - 1;
    double delta_lin = ep->delta0 * D;

    // printf("delta=%f\t %f\n", delta, delta_lin);

    /* Evaluate the elliptic integrals */
    double b[3] = {1,1,1};

    double x[3];
    double x_freeze = ep->f_collapse * a;

    for (int i=0; i<3; i++) {
        x[i] = y[i] > x_freeze ? y[i] : x_freeze;
    }

    double rt = cbrt(x[0]*x[1]*x[2]);

    for (int i=0; i<3; i++) {
        b[i] = gsl_sf_ellint_RJ(x[0]*x[0]/(rt*rt), x[1]*x[1]/(rt*rt), x[2]*x[2]/(rt*rt), x[i]*x[i]/(rt*rt), GSL_PREC_DOUBLE);
    }

    // printf("t=%f\t a=%e\t addot=%e\t delta = %e\t %f\t D=%f\t OmH=%e\n", t, a, addot, delta, delta_lin, D, OmH);
    // printf("%f %f %f %f\n", c[0], c[1], c[2], c[0]);

    // printf("%e\n", delta);

    /* x[0:3] = y[0:3], x_dot[0:3] = y[3:6] */

    for (int i=0; i<3; i++) {
        if (y[i] <= x_freeze) {
            f[i] = 0;
            f[i+3] = 0;
        } else {
            f[i] = y[i+3];
            f[i+3] = y[i] * (addot / a - 0.5 * OmH2 * (b[i] * delta + c[i] * delta_lin));
        }
    }

    // int i=0;
    // printf("%e %e %e\n", addot / a,  - 0.5 * OmH2 * (b[i] * delta),    - 0.5 * OmH2 *(c[i] * delta_lin));

    // f[0] = y[3];
    // f[1] = y[4];
    // f[2] = y[5];
    //
    // f[4] = y[1] * (addot / a - 0.5 * OmH * (b[1] * delta + c[1] * delta_lin));
    // f[5] = y[2] * (addot / a - 0.5 * OmH * (b[2] * delta + c[2] * delta_lin));

    // printf("%e\n", y[0] * (addot / a - 0.5 * OmH * (b[0] * delta + c[0] * delta_lin)));
    // printf("%e\n", y[1] * (addot / a - 0.5 * OmH * (b[1] * delta + c[1] * delta_lin)));
    // printf("%e\n", y[2] * (addot / a - 0.5 * OmH * (b[2] * delta + c[2] * delta_lin)));

    return GSL_SUCCESS;
}


int run_meshpt(int N, double boxlen, void *gridv, int nk, void *kvecv,
               void *sqrtPvecv, int nz, void *tvecv, void *Dvecv, void *avecv,
               void *addotvecv, void *OmH2vecv, void *Hvecv, int N_SPT, double t_ini, double t_final,
               double k_cutoff, char *output_dir, int fast_EdS_mode) {

    /* The output grid */
    double *grid = (double *)gridv;

    /* Memory block for the input data */
    double *kvec = (double *)kvecv;
    double *sqrtPvec = (double *)sqrtPvecv;
    double *tvec = (double *)tvecv;
    double *Dvec = (double *)Dvecv;
    double *avec = (double *)avecv;
    double *addotvec = (double *)addotvecv;
    double *OmH2vec = (double *)OmH2vecv;
    double *Hvec = (double *)Hvecv;

    /* Initialize power spectrum interpolation spline */
    struct strooklat Pspline = {kvec, nk};
    init_strooklat_spline(&Pspline, 100);

    /* Initialize a spline for the time variable (log of growth factor D) */
    struct strooklat spline = {tvec, nz};
    init_strooklat_spline(&spline, 100);

    double t0 = t_ini, t1 = t_final;

    double a0 = strooklat_interp(&spline, avec, t0);
    double H0 = strooklat_interp(&spline, Hvec, t0);
    double D0 = strooklat_interp(&spline, Dvec, t0);


    double e = 0.1;
    double p = 0.07;

    double c1 = p - 3*e;
    double c2 = -2*p;
    double c3 = p + 3*e;

    double delta_bar = 4.13;

    struct ellipt_params ep = {delta_bar, {c1, c2, c3}, Dvec, avec, addotvec, OmH2vec, 0.171, &spline};



    // double y = gsl_sf_ellint_RJ(0.031, 0.031, 0.031, 0.031, GSL_PREC_DOUBLE);
    // printf ("int = %e\n",y);

    gsl_odeiv2_system sys = {f, NULL, 6, &ep};

    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(
                               &sys, gsl_odeiv2_step_rkf45, 1e-3, 1e-3, 0.0);
    int i;


    double x0 = a0;


    double lambda1 = (D0*delta_bar/3) * (1 + c1);
    double lambda2 = (D0*delta_bar/3) * (1 + c2);
    double lambda3 = (D0*delta_bar/3) * (1 + c3);

    double x1 = 1 * (1-lambda1);
    double x2 = 1 * (1-lambda2);
    double x3 = 1 * (1-lambda3);

    /* Determine velocities with finite differences */
    double t_plus = t_ini * 1.001;
    double a_plus = strooklat_interp(&spline, avec, t_plus);
    double H_plus = strooklat_interp(&spline, Hvec, t_plus);
    double D_plus = strooklat_interp(&spline, Dvec, t_plus);

    double lambda1_plus = (D_plus*delta_bar/3) * (1 + c1);
    double lambda2_plus = (D_plus*delta_bar/3) * (1 + c2);
    double lambda3_plus = (D_plus*delta_bar/3) * (1 + c3);

    double x1_plus = 1 * (1-lambda1_plus);
    double x2_plus = 1 * (1-lambda2_plus);
    double x3_plus = 1 * (1-lambda3_plus);

    double v1 = (x1_plus - x1)/(t_plus - t_ini);
    double v2 = (x2_plus - x2)/(t_plus - t_ini);
    double v3 = (x3_plus - x3)/(t_plus - t_ini);

    double u1 = -1 * H0 * lambda1;
    double u2 = -1 * H0 * lambda2;
    double u3 = -1 * H0 * lambda3;

    // v1 = v2 = v3 = 0;

    printf("a0 = %f\n", a0);
    printf("x = %f %f %f\n", x1, x2, x3);
    printf("y = %f %f %f\n", x1_plus, x2_plus, x3_plus);
    printf("v = %f %f %f\n", v1, v2, v3);
    printf("u = %f %f %f\n", u1, u2, u3);
    printf("delta = %f\n", (a0*a0*a0/(x1*x2*x3)-1));


    double y[6] = {x1, x2, x3, v1, v2, v3};
    double t = t0;

    for (i = 1; i <= 100; i++) {
        double a = strooklat_interp(&spline, avec, t);
        double delta = a*a*a/(y[0]*y[1]*y[2])-1;

        printf("%.5e %.5e %.5e %.5e %.5e\n", delta, a, y[0]/x1, y[1]/x2, y[2]/x3);

        double ti = t0 + (t1-t0) * i * 1.0 / 100.0;
        int status = gsl_odeiv2_driver_apply(d, &t, ti, y);

        if (status != GSL_SUCCESS) {
            printf("error, return value=%d\n", status);
            break;
        }

        double x_freeze = ep.f_collapse * a;

        if (y[0] <= x_freeze && y[1] <= x_freeze && y[2] <= x_freeze)
            break;


    }

    gsl_odeiv2_driver_free(d);

    printf("HUH %e %e %e\n", strooklat_interp(&spline, avec, t0), tvec[nz-1], avec[nz-1]);

    return;

    // /* Index table lengths */
    // int min_length = 10000; // number of coefficients
    // int cache_length = 4;
    // int timesteps = 100;
    //
    // /* Initialize the random number generator */
    // int s = 101;
    // rng_state seed = rand_uint64_init(s);
    //
    // printf("The output diretory is '%s'.\n", output_dir);
    //
    // /* A unique number to prevent filename clashes */
    // int unique = (int)(sampleUniform(&seed) * 1e6);
    //
    // /* Allocate array for the primordial Gaussian field */
    // fftw_complex *fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    // double *box = malloc(N * N * N * sizeof(double));
    //
    // /* Generate a complex Hermitian Gaussian random field */
    // generate_complex_grf(fbox, N, boxlen, &seed);
    // enforce_hermiticity(fbox, N, boxlen);
    //
    // /* Apply the interpolated power spectrum to the Gaussian field */
    // struct spline_params sp = {&Pspline, sqrtPvec};
    // fft_apply_kernel(fbox, fbox, N, boxlen, kernel_sqrt_power_spline, &sp);
    //
    // /* Apply a k-cutoff to address UV divergences */
    // if (k_cutoff > 0) {
    //     printf("The cutoff is %f\n", k_cutoff);
    //     fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_cutoff);
    // }
    //
    // /* Fourier transform the grid */
    // fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    // fft_execute(c2r);
    // fft_normalize_c2r(box, N, boxlen);
    // fftw_destroy_plan(c2r);
    //
    // /* Free the complex grid */
    // fftw_free(fbox);
    //
    // memcpy(grid, box, N * N * N * sizeof(double));
    //
    // /* Free the real grid */
    // fftw_free(box);
    //
    // /* Free the splines */
    // free_strooklat_spline(&Pspline);
    // free_strooklat_spline(&spline);

    return 0;
}

int main() {
    printf("Nice try.\n");

    return 0;
}
