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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

#include "../include/peak.h"

typedef int (*func_ptr) (void * params, ...);

int run_meshpt(int N, double boxlen, void *gridv, int nk, void *kvecv,
               void *sqrtPvecv, int nz, void *logDvecv, void *Omega21v,
               void *Omega22v, int N_SPT, double D_ini, double D_final,
               double k_cutoff, char *output_dir, int fast_EdS_mode) {

    /* The output grid */
    double *grid = (double *)gridv;

    /* Memory block for the input data */
    double *kvec = (double *)kvecv;
    double *sqrtPvec = (double *)sqrtPvecv;
    double *logDvec = (double *)logDvecv;
    double *Omega_21vec = (double *)Omega21v;
    double *Omega_22vec = (double *)Omega22v;

    /* Initialize power spectrum interpolation spline */
    struct strooklat Pspline = {kvec, nk};
    init_strooklat_spline(&Pspline, 100);

    /* Initialize a spline for the time variable (log of growth factor D) */
    struct strooklat spline = {logDvec, nz};
    init_strooklat_spline(&spline, 100);

    /* Index table lengths */
    int min_length = 10000; // number of coefficients
    int cache_length = 4;
    int timesteps = 100;

    /* Starting and ending times */
    double t_i = log(D_ini);
    double t_f = log(D_final);

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
    if (k_cutoff > 0) {
        printf("The cutoff is %f\n", k_cutoff);
        fft_apply_kernel(fbox, fbox, N, boxlen, kernel_lowpass, &k_cutoff);
    }

    /* Fourier transform the grid */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(box, N, boxlen);
    fftw_destroy_plan(c2r);

    /* Free the complex grid */
    fftw_free(fbox);

    memcpy(grid, box, N*N*N*sizeof(double));

    /* Free the real grid */
    fftw_free(box);

    /* Free the splines */
    free_strooklat_spline(&Pspline);
    free_strooklat_spline(&spline);

    return 0;
}

int main() {
    printf("Nice try.\n");

    return 0;
}
