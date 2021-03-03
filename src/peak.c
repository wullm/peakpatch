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

#include <gsl/gsl_eigen.h>
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
    double f_collapse[3];
    struct strooklat *spline;
};

int mean_stress(int N, int x0, int y0, int z0, int R, double *box,
                double *psi_xx, double *psi_xy, double *psi_xz, double *psi_yy,
                double *psi_yz, double *psi_zz, double *lambda_1,
                double *lambda_2, double *lambda_3, double *delta_bar) {

    double stress[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double avg_delta = 0;

    int count = 0;

    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(3);
    gsl_vector *eval = gsl_vector_alloc(3);
    gsl_matrix *evec = gsl_matrix_alloc(3, 3);

    for (int x = -R; x <= R; x++) {
        for (int y = -R; y <= R; y++) {
            for (int z = -R; z <= R; z++) {

                int id = row_major(x0 + x, y0 + y, z0 + z, N);
                stress[0] += psi_xx[id];
                stress[1] += psi_xy[id];
                stress[2] += psi_xz[id];
                stress[3] += psi_xy[id];
                stress[4] += psi_yy[id];
                stress[5] += psi_yz[id];
                stress[6] += psi_xz[id];
                stress[7] += psi_yz[id];
                stress[8] += psi_zz[id];

                avg_delta += box[id];

                count++;
            }
        }
    }

    for (int i = 0; i < 9; i++) {
        stress[i] /= count;
    }
    avg_delta /= count;

    gsl_matrix_view m = gsl_matrix_view_array(stress, 3, 3);
    gsl_eigen_symmv(&m.matrix, eval, evec, w);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);

    *lambda_1 = gsl_vector_get(eval, 0);
    *lambda_2 = gsl_vector_get(eval, 1);
    *lambda_3 = gsl_vector_get(eval, 2);
    *delta_bar = avg_delta;

    gsl_vector_free(eval);
    gsl_matrix_free(evec);

    return 0;
}

int f(double t, const double y[], double f[], void *params) {
    struct ellipt_params *ep = (struct ellipt_params *)params;

    double OmH2 = strooklat_interp(ep->spline, ep->OmH2_vec, t);
    double D = strooklat_interp(ep->spline, ep->D_vec, t);
    double a = strooklat_interp(ep->spline, ep->a_vec, t);
    double addot = strooklat_interp(ep->spline, ep->addot_vec, t);
    double *c = ep->c;

    // OmH2 = 0.00145;

    double delta = a * a * a / (y[0] * y[1] * y[2]) - 1;
    double delta_lin = ep->delta0 * D;

    // printf("delta=%f\t %f\n", delta, delta_lin);

    /* Evaluate the elliptic integrals */
    double b[3] = {1, 1, 1};

    double x[3];

    for (int i = 0; i < 3; i++) {
        x[i] = y[i] > ep->f_collapse[i] * a ? y[i] : ep->f_collapse[i] * a;
    }

    double rt = cbrt(x[0] * x[1] * x[2]);

    for (int i = 0; i < 3; i++) {
        b[i] = gsl_sf_ellint_RJ(
                   x[0] * x[0] / (rt * rt), x[1] * x[1] / (rt * rt),
                   x[2] * x[2] / (rt * rt), x[i] * x[i] / (rt * rt), GSL_PREC_SINGLE);
    }

    // printf("t=%f\t a=%e\t addot=%e\t delta = %e\t %f\t D=%f\t OmH=%e\n", t,
    // a, addot, delta, delta_lin, D, OmH); printf("%f %f %f %f\n", c[0], c[1],
    // c[2], c[0]);

    // printf("%e\n", delta);

    /* x[0:3] = y[0:3], x_dot[0:3] = y[3:6] */

    for (int i = 0; i < 3; i++) {
        if (y[i] <= ep->f_collapse[i] * a) {
            f[i] = 0;
            f[i + 3] = 0;
        } else {
            f[i] = y[i + 3];
            f[i + 3] = y[i] * (addot / a -
                               0.5 * OmH2 * (b[i] * delta + c[i] * delta_lin));
        }
    }

    // int i=0;
    // printf("%e %e %e\n", addot / a,  - 0.5 * OmH2 * (b[i] * delta),    - 0.5
    // * OmH2 *(c[i] * delta_lin));

    // f[0] = y[3];
    // f[1] = y[4];
    // f[2] = y[5];
    //
    // f[4] = y[1] * (addot / a - 0.5 * OmH * (b[1] * delta + c[1] *
    // delta_lin)); f[5] = y[2] * (addot / a - 0.5 * OmH * (b[2] * delta + c[2]
    // * delta_lin));

    // printf("%e\n", y[0] * (addot / a - 0.5 * OmH * (b[0] * delta + c[0] *
    // delta_lin))); printf("%e\n", y[1] * (addot / a - 0.5 * OmH * (b[1] *
    // delta + c[1] * delta_lin))); printf("%e\n", y[2] * (addot / a - 0.5 * OmH
    // * (b[2] * delta + c[2] * delta_lin)));

    return GSL_SUCCESS;
}

double step_down(struct ellipt_params *ep, double t_ini, double t_max) {
    double a0 = strooklat_interp(ep->spline, ep->a_vec, t_ini);
    double D0 = strooklat_interp(ep->spline, ep->D_vec, t_ini);

    double x0 = a0;

    double lambda1 = (D0 * ep->delta0 / 3) * (1 + ep->c[0]);
    double lambda2 = (D0 * ep->delta0 / 3) * (1 + ep->c[1]);
    double lambda3 = (D0 * ep->delta0 / 3) * (1 + ep->c[2]);

    double x1 = 1 * (1 - lambda1);
    double x2 = 1 * (1 - lambda2);
    double x3 = 1 * (1 - lambda3);

    /* Determine velocities with finite differences */
    double t_plus = t_ini * 1.001;
    double a_plus = strooklat_interp(ep->spline, ep->a_vec, t_plus);
    double D_plus = strooklat_interp(ep->spline, ep->D_vec, t_plus);

    double lambda1_plus = (D_plus * ep->delta0 / 3) * (1 + ep->c[0]);
    double lambda2_plus = (D_plus * ep->delta0 / 3) * (1 + ep->c[1]);
    double lambda3_plus = (D_plus * ep->delta0 / 3) * (1 + ep->c[2]);

    double x1_plus = 1 * (1 - lambda1_plus);
    double x2_plus = 1 * (1 - lambda2_plus);
    double x3_plus = 1 * (1 - lambda3_plus);

    double v1 = (x1_plus - x1) / (t_plus - t_ini);
    double v2 = (x2_plus - x2) / (t_plus - t_ini);
    double v3 = (x3_plus - x3) / (t_plus - t_ini);

    // printf("a0 = %f\n", a0);
    printf("x = %f %f %f\n", x1, x2, x3);
    // printf("y = %f %f %f\n", x1_plus, x2_plus, x3_plus);
    // printf("v = %f %f %f\n", v1, v2, v3);
    // printf("delta = %f\n", (a0*a0*a0/(x1*x2*x3)-1));

    double y[6] = {x1, x2, x3, v1, v2, v3};
    double t = t_ini;

    double delta_t = 1.001;
    double a = 0;

    gsl_odeiv2_system sys = {f, NULL, 6, ep};

    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(
                               &sys, gsl_odeiv2_step_rkf45, 1e-3, 1e-3, 0.0);

    while ((y[0] > ep->f_collapse[0] * a || y[1] > ep->f_collapse[1] * a ||
            y[2] > ep->f_collapse[2] * a) &&
            t < t_max) {
        a = strooklat_interp(ep->spline, ep->a_vec, t);
        // double delta = a*a*a/(y[0]*y[1]*y[2])-1;
        // printf("%.5e %.5e %.5e %.5e %.5e\n", delta, a, y[0]/x1, y[1]/x2,
        // y[2]/x3);

        double ti = t * delta_t;
        int status = gsl_odeiv2_driver_apply(d, &t, ti, y);

        if (status != GSL_SUCCESS) {
            printf("error, return value=%d\n", status);
            break;
        }
    }

    printf("t tmax %f %f\n", t, t_max);

    gsl_odeiv2_driver_free(d);

    return a;
}

int run_meshpt(int N, double boxlen, void *gridv, int nk, void *kvecv,
               void *sqrtPvecv, int nz, void *tvecv, void *Dvecv, void *avecv,
               void *addotvecv, void *OmH2vecv, void *Hvecv, int N_SPT,
               double t_ini, double t_final, double k_cutoff, char *output_dir,
               int fast_EdS_mode) {

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

    /* Index table lengths */
    int min_length = 10000; // number of coefficients
    int cache_length = 4;
    int timesteps = 100;

    /* Initialize the random number generator */
    int s = 101;
    rng_state seed = rand_uint64_init(s);

    printf("The output diretory is '%s'.\n", output_dir);

    /* A unique number to prevent filename clashes */
    int unique = (int)(sampleUniform(&seed) * 1e6);

    /* Allocate array for the primordial Gaussian field */
    fftw_complex *fbox = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    double *box = malloc(N * N * N * sizeof(double));

    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(fbox, N, boxlen, &seed);
    enforce_hermiticity(fbox, N, boxlen);

    /* Apply the interpolated power spectrum to the Gaussian field */
    struct spline_params sp = {&Pspline, sqrtPvec};
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_sqrt_power_spline, &sp);

    /* Apply a k-cutoff to address UV divergences */
    double R_smooth = 2 * M_PI / k_cutoff;
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_real_tophat, &R_smooth);

    /* Allocate memory for the stress tensor grids */
    fftw_complex *f_kern = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_xx = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_xy = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_xz = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_yy = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_yz = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    fftw_complex *f_psi_zz = malloc(N * N * (N / 2 + 1) * sizeof(fftw_complex));
    double *psi_xx = malloc(N * N * N * sizeof(double));
    double *psi_xy = malloc(N * N * N * sizeof(double));
    double *psi_xz = malloc(N * N * N * sizeof(double));
    double *psi_yy = malloc(N * N * N * sizeof(double));
    double *psi_yz = malloc(N * N * N * sizeof(double));
    double *psi_zz = malloc(N * N * N * sizeof(double));

    /* Approximate the potential with the Zel'dovich approximation */
    fft_apply_kernel(f_kern, fbox, N, boxlen, kernel_inv_poisson, NULL);

    /* Compute the displacements grids by differentiating the potential */
    fft_apply_kernel(f_psi_xx, f_kern, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_xx, f_psi_xx, N, boxlen, kernel_dx, NULL);

    fft_apply_kernel(f_psi_xy, f_kern, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_xy, f_psi_xy, N, boxlen, kernel_dy, NULL);

    fft_apply_kernel(f_psi_xz, f_kern, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(f_psi_xz, f_psi_xz, N, boxlen, kernel_dz, NULL);

    fft_apply_kernel(f_psi_yy, f_kern, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(f_psi_yy, f_psi_yy, N, boxlen, kernel_dy, NULL);

    fft_apply_kernel(f_psi_yz, f_kern, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(f_psi_yz, f_psi_yz, N, boxlen, kernel_dz, NULL);

    fft_apply_kernel(f_psi_zz, f_kern, N, boxlen, kernel_dz, NULL);
    fft_apply_kernel(f_psi_zz, f_psi_zz, N, boxlen, kernel_dz, NULL);

    /* Fourier transform the potential grids */
    fftw_plan c2r_xx =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_xx, psi_xx, FFTW_ESTIMATE);
    fftw_plan c2r_xy =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_xy, psi_xy, FFTW_ESTIMATE);
    fftw_plan c2r_xz =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_xz, psi_xz, FFTW_ESTIMATE);
    fftw_plan c2r_yy =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_yy, psi_yy, FFTW_ESTIMATE);
    fftw_plan c2r_yz =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_yz, psi_yz, FFTW_ESTIMATE);
    fftw_plan c2r_zz =
        fftw_plan_dft_c2r_3d(N, N, N, f_psi_zz, psi_zz, FFTW_ESTIMATE);

    fft_execute(c2r_xx);
    fft_execute(c2r_xy);
    fft_execute(c2r_xz);
    fft_execute(c2r_yy);
    fft_execute(c2r_yz);
    fft_execute(c2r_zz);

    fft_normalize_c2r(psi_xx, N, boxlen);
    fft_normalize_c2r(psi_xy, N, boxlen);
    fft_normalize_c2r(psi_xz, N, boxlen);
    fft_normalize_c2r(psi_yy, N, boxlen);
    fft_normalize_c2r(psi_yz, N, boxlen);
    fft_normalize_c2r(psi_zz, N, boxlen);

    fftw_destroy_plan(c2r_xx);
    fftw_destroy_plan(c2r_xy);
    fftw_destroy_plan(c2r_xz);
    fftw_destroy_plan(c2r_yy);
    fftw_destroy_plan(c2r_yz);
    fftw_destroy_plan(c2r_zz);

    fftw_free(f_kern);
    fftw_free(f_psi_xx);
    fftw_free(f_psi_xy);
    fftw_free(f_psi_xz);
    fftw_free(f_psi_yy);
    fftw_free(f_psi_yz);
    fftw_free(f_psi_zz);

    /* Fourier transform the grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(box, N, boxlen);
    fftw_destroy_plan(c2r);

    /* Free the complex grid */
    fftw_free(fbox);

    memcpy(grid, box, N * N * N * sizeof(double));

    // for (int x=0; x<N; x++) {
    //     for (int y=0; y<N; y++) {
    //         for (int z=0; z<N; z++) {
    //             int id = row_major(x, y, z, N);
    //             double stress[] = {psi_xx[id], psi_xy[id], psi_xz[id],
    //                                psi_xy[id], psi_yy[id], psi_yz[id],
    //                                psi_xz[id], psi_yz[id], psi_zz[id]};
    //             gsl_matrix_view m = gsl_matrix_view_array(stress, 3, 3);
    //             gsl_eigen_symmv(&m.matrix, eval, evec, w);
    //             gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);
    //
    //             double lambda_1 = gsl_vector_get(eval, 0);
    //             double lambda_2 = gsl_vector_get(eval, 1);
    //             double lambda_3 = gsl_vector_get(eval, 2);
    //             printf ("eigenvalue = %f %f %f\n", lambda_1, lambda_2,
    //             lambda_3);
    //
    //         }
    //     }
    // }

    double lambda_1, lambda_2, lambda_3;
    double delta;

    int R = ceil(R_smooth / boxlen * N);

    /* Find candidate peaks (delta > 1.5 & local max)*/
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                int id = row_major(x, y, z, N);
                double d = box[id];

                if (d > 1.5) {
                    /* Check if it is a local max */
                    char local_max = 1;
                    for (int i=-1; i<=1; i++) {
                        for (int j=-1; j<=1; j++) {
                            for (int k=-1; k<=1; k++) {
                                if (d < box[row_major(x+i, y+j, z+k, N)])
                                    local_max = 0;
                            }
                        }
                    }

                    if (!local_max) break;

                    printf("Candidate %d %d %d %f %d\n", x, y, z, d, R);

                    mean_stress(N, x, y, z, 0, box, psi_xx, psi_xy, psi_xz, psi_yy, psi_yz,
                                psi_zz, &lambda_1, &lambda_2, &lambda_3, &delta);

                    double delta_alt = lambda_1 + lambda_2 + lambda_3;
                    printf("eigenvalue = %f %f %f\n", lambda_1, lambda_2, lambda_3);
                    printf("avg density %f %f\n", delta, delta_alt);

                    double c1 = -lambda_1 / (delta / 3) + 1;
                    double c2 = -lambda_2 / (delta / 3) + 1;
                    double c3 = -lambda_3 / (delta / 3) + 1;

                    printf("c[3] = %f %f %f %f\n", c1, c2, c3, c1 + c2 + c3);

                    double p = -c2 / 2;
                    double e = (c1 - p) / 3;
                    double e_alt = -(c3 - p) / 3;

                    printf("p = %f\n", p);
                    printf("e = %f\n", e);
                    printf("e = %f\n", e_alt);

                    // e = 0.1;
                    // p = 0.05;
                    //
                    // c1 = p - 3*e;
                    // c2 = -2*p;
                    // c3 = p + 3*e;
                    //
                    // delta = 2.13;

                    struct ellipt_params ep = {
                        delta,   {c1, c2, c3},         Dvec,   avec, addotvec,
                        OmH2vec, {0.01, 0.171, 0.171}, &spline
                    };

                    double a_collapse = step_down(&ep, t_ini, t_final);

                    printf("a_collapse = %f\n", a_collapse);
                }
            }
        }
    }




    /* Free the real grid */
    fftw_free(box);
    fftw_free(psi_xx);
    fftw_free(psi_xy);
    fftw_free(psi_xz);
    fftw_free(psi_yy);
    fftw_free(psi_yz);
    fftw_free(psi_zz);

    /* Free the splines */
    free_strooklat_spline(&Pspline);
    free_strooklat_spline(&spline);

    return 0;
}

int main() {
    printf("Nice try.\n");

    return 0;
}
