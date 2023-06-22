#include "ising.h"
#include <emmintrin.h>
#include <xmmintrin.h>
#include <math.h>
#include <stdlib.h>

void update(const float temp, int grid[L][L])
{
    const __m128i zero = _mm_setzero_si128();

    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 4) {
            __m128i spin_old = _mm_loadu_si128((__m128i*)&grid[i][j]);
            __m128i spin_new = _mm_sub_epi32(zero, spin_old);

            // computing energy contributions of neighbors
            __m128i spin_neigh_n = _mm_loadu_si128((__m128i*)&grid[(i + L - 1) % L][j]);
            __m128i spin_neigh_e = _mm_loadu_si128((__m128i*)&grid[i][(j + 1) % L]);
            __m128i spin_neigh_w = _mm_loadu_si128((__m128i*)&grid[i][(j + L - 1) % L]);
            __m128i spin_neigh_s = _mm_loadu_si128((__m128i*)&grid[(i + 1) % L][j]);

            __m128i delta_spin_neigh_n = _mm_sub_epi32(spin_neigh_n, spin_old);
            __m128i delta_spin_neigh_e = _mm_sub_epi32(spin_neigh_e, spin_old);
            __m128i delta_spin_neigh_w = _mm_sub_epi32(spin_neigh_w, spin_old);
            __m128i delta_spin_neigh_s = _mm_sub_epi32(spin_neigh_s, spin_old);

            // computing h_before and h_after
            __m128i h_before = _mm_mullo_epi32(spin_old, _mm_add_epi32(delta_spin_neigh_n, _mm_add_epi32(delta_spin_neigh_e, _mm_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s))));
            __m128i h_after = _mm_mullo_epi32(spin_new, _mm_add_epi32(delta_spin_neigh_n, _mm_add_epi32(delta_spin_neigh_e, _mm_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s))));

            __m128i delta_E = _mm_sub_epi32(h_after, h_before);

            // Generate random numbers
            __m128i rand_num = _mm_set_epi32(rand(), rand(), rand(), rand());
            __m128i rand_max = _mm_set1_epi32(RAND_MAX);
            __m128 p = _mm_cvtepi32_ps(_mm_div_epi32(rand_num, rand_max));

            __m128 exp_delta_E = exp_ps(_mm_mul_ps(_mm_set1_ps(-1.0f / temp), _mm_cvtepi32_ps(delta_E)));

            __m128 mask = _mm_cmple_ps(delta_E, _mm_setzero_ps());
            mask = _mm_or_ps(mask, _mm_cmple_ps(p, exp_delta_E));

            __m128i updated_spin = _mm_or_si128(_mm_and_si128(mask, spin_new), _mm_andnot_si128(mask, spin_old));

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
