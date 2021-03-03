/*******************************************************************************
 * This file is part of Mitos.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
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

#include "../include/grf.h"
#include "../include/fft.h"
#include "../include/peak.h"

#include <math.h>

int generate_complex_grf(fftw_complex *fbox, int N, double boxlen,
                         rng_state *state) {
    /* The complex array is N * N * (N/2 + 1), locally we have NX * N * (N/2 +
     * 1) */
    const double boxvol = boxlen * boxlen * boxlen;
    const double factor = sqrt(boxvol / 2);
    const double dk = 2 * M_PI / boxlen;

    /* Refer to fourier.pdf for details. */

    /* Because the Gaussian field is real, the Fourier transform fbox
     * is Hermitian. This can be stored with just N*N*(N/2+1) complex
     * numbers. */

    double kx, ky, kz, k;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z <= N / 2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Ignore the constant DC mode */
                if (k > 0) {
                    double a = sampleNorm(state);
                    double b = sampleNorm(state);
                    double norm = sqrt(a * a + b * b);
                    fbox[row_major_half(x, y, z, N)] =
                        -sqrt(2) * (a + b * I) / norm * factor;
                } else {
                    fbox[row_major_half(x, y, z, N)] = 0;
                }
            }
        }
    }

    return 0;
}

/* Perform corrections to the generated Gaussian random field such that the
 * complex array is truly Hermitian. This only affects the planes k_z = 0
 * and k_z = N/2.  */
int enforce_hermiticity(fftw_complex *fbox, int N, double boxlen) {
    /* The first (k=0) and last (k=N/2+1) planes need hermiticity enforced */

    /* For both planes */
    for (int z = 0; z <= N / 2; z += N / 2) { // runs over z=0 and z=N/2

        /* Enforce hermiticity: f(k) = f*(-k) */
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                if (x > N / 2)
                    continue; // skip the upper half
                if ((x == 0 || x == N / 2) && y > N / 2)
                    continue; // skip two strips

                int invx = (x > 0) ? N - x : 0;
                int invy = (y > 0) ? N - y : 0;
                int invz = (z > 0) ? N - z : 0; // maps 0->0 and (N/2)->(N/2)

                int id = row_major_half(x, y, z, N);
                int invid = row_major_half(invx, invy, invz, N);

                /* If the point maps to itself, throw away the imaginary part */
                if (invx == x && invy == y && invz == z) {
                    fbox[id] = creal(fbox[id]);
                } else {
                    /* Otherwise, set it to the conjugate of its mirror point */
                    fbox[id] = conj(fbox[invid]);
                }
            }
        }
    }

    return 0;
}
