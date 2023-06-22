#include "ising.h"

#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

//Modificada Comentada
#include <xmmintrin.h> // SSE intrinsics

void update(const float temp, int grid[L][L])
{
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 4) {
            // Load data into SSE registers
            __m128i spin_old = _mm_loadu_si128((__m128i*)&grid[i][j]);
            __m128i spin_new = _mm_sub_epi32(_mm_set1_epi32(-1), spin_old);

            // Compute energy contributions of neighbors
            __m128i spin_neigh_n = _mm_loadu_si128((__m128i*)&grid[(i + 1) % L][j]);
            __m128i spin_neigh_e = _mm_loadu_si128((__m128i*)&grid[i][(j + 1) % L]);
            __m128i spin_neigh_w = _mm_loadu_si128((__m128i*)&grid[i][(j + L - 1) % L]);
            __m128i spin_neigh_s = _mm_loadu_si128((__m128i*)&grid[(i + L - 1) % L][j]);

            __m128i delta_spin_neigh_n = _mm_sub_epi32(spin_neigh_n, spin_old);
            __m128i delta_spin_neigh_e = _mm_sub_epi32(spin_neigh_e, spin_old);
            __m128i delta_spin_neigh_w = _mm_sub_epi32(spin_neigh_w, spin_old);
            __m128i delta_spin_neigh_s = _mm_sub_epi32(spin_neigh_s, spin_old);

            // Compute h_before and h_after
            __m128i h_before = _mm_add_epi32(_mm_add_epi32(delta_spin_neigh_n, delta_spin_neigh_e),
                                              _mm_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s));
            __m128i h_after = _mm_sub_epi32(h_before, _mm_slli_epi32(spin_new, 2));

            __m128i delta_E = _mm_sub_epi32(h_after, h_before);
            __m128 p = _mm_div_ps(_mm_cvtepi32_ps(_mm_set1_epi32(rand())), _mm_set1_ps(RAND_MAX));

            // Compare delta_E with zero and p with exp(-delta_E / temp)
            __m128 accept_mask = _mm_or_ps(_mm_cmple_ps(_mm_setzero_ps(), _mm_cvtepi32_ps(delta_E)),
                                            _mm_cmple_ps(p, _mm_exp_ps(_mm_div_ps(_mm_cvtepi32_ps(delta_E), _mm_set1_ps(temp)))));

            // Blend spin_old and spin_new based on accept_mask
            __m128i result = _mm_blendv_epi8(spin_old, spin_new, _mm_castps_si128(accept_mask));

            // Store the result back to memory
            _mm_storeu_si128((__m128i*)&grid[i][j], result);
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
