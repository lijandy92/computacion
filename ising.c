

//Modificada Comentada
#include "ising.h"
#include <math.h>
#include <stdlib.h>
#include <xmmintrin.h>

void update(const float temp, int grid[L][L])
{
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 4) {
            // Load data from grid into SIMD registers
            __m128i spin_old = _mm_loadu_si128((__m128i*)&grid[i][j]);
            __m128i spin_new = _mm_sub_epi32(_mm_setzero_si128(), spin_old);

            // Compute energy contributions of neighbors
            __m128i spin_neigh_n = _mm_loadu_si128((__m128i*)&grid[(i + L - 1) % L][j]);
            __m128i spin_neigh_e = _mm_loadu_si128((__m128i*)&grid[i][(j + 1) % L]);
            __m128i spin_neigh_w = _mm_loadu_si128((__m128i*)&grid[i][(j + L - 1) % L]);
            __m128i spin_neigh_s = _mm_loadu_si128((__m128i*)&grid[(i + 1) % L][j]);

            __m128i delta_spin_neigh_n = _mm_sub_epi32(spin_neigh_n, spin_old);
            __m128i delta_spin_neigh_e = _mm_sub_epi32(spin_neigh_e, spin_old);
            __m128i delta_spin_neigh_w = _mm_sub_epi32(spin_neigh_w, spin_old);
            __m128i delta_spin_neigh_s = _mm_sub_epi32(spin_neigh_s, spin_old);

            // Compute h_before and h_after
            __m128i h_before = _mm_mullo_epi32(_mm_set1_epi32(-1), _mm_add_epi32(_mm_add_epi32(delta_spin_neigh_n, delta_spin_neigh_e), _mm_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s)));
            __m128i h_after = _mm_mullo_epi32(_mm_sub_epi32(_mm_setzero_si128(), spin_new), _mm_add_epi32(_mm_add_epi32(delta_spin_neigh_n, delta_spin_neigh_e), _mm_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s)));

            // Compute delta_E
            __m128i delta_E = _mm_sub_epi32(h_after, h_before);

            // Generate random numbers
            __m128 p = _mm_set_ps((float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);

            // Compare delta_E with zero
            __m128i mask_delta_E = _mm_cmple_epi32(delta_E, _mm_setzero_si128());

            // Compute expf(-delta_E / temp)
            __m128 exp_delta_E_temp = _mm_div_ps(_mm_set1_ps(1.0f), _mm_exp_ps(_mm_div_ps(_mm_cvtepi32_ps(delta_E), _mm_set1_ps(temp))));

            // Compare p with exp_delta_E_temp
            __m128 mask_p = _mm_cmple_ps(p, exp_delta_E_temp);

            // Mask grid with spin_new based on comparison results
            __m128i masked_spin_new = _mm_and_si128(_mm_castps_si128(mask_p), spin_new);
            __m128i masked_spin_old = _mm_andnot_si128(_mm_castps_si128(mask_p), spin_old);
            __m128i updated_spin = _mm_or_si128(masked_spin_new, masked_spin_old);

            // Store updated_spin back to grid
            _mm_storeu_si128((__m128i*)&grid[i][j], updated_spin);
        }
    }
}



//Modificada
double calculate(int grid[L][L], int* M_max)
{
    int E = 0;
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            int spin = grid[i][j];
            int spin_neigh_n = grid[(i + 1) % L][j];
            int spin_neigh_e = grid[i][(j + 1) % L];
            int spin_neigh_w = grid[i][(j + L - 1) % L];
            int spin_neigh_s = grid[(i + L - 1) % L][j];

            E += (spin * spin_neigh_n) + (spin * spin_neigh_e) + (spin * spin_neigh_w) + (spin * spin_neigh_s);
            *M_max += spin;
        }
    }
    return -((double)E / 2.0);
}
