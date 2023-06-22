#include "ising.h"

#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

//Modificada Comentada
void update(const float temp, int grid[L][L])
{
    // typewriter update
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 8) {
            __m256i spin_old = _mm256_loadu_si256((__m256i*)&grid[i][j]);
            __m256i spin_new = _mm256_sub_epi32(_mm256_set1_epi32(-1), spin_old);

            // computing energy contributions of neighbors
            __m256i spin_neigh_n = _mm256_loadu_si256((__m256i*)&grid[(i + L - 1) % L][j]);
            __m256i spin_neigh_e = _mm256_loadu_si256((__m256i*)&grid[i][(j + 1) % L]);
            __m256i spin_neigh_w = _mm256_loadu_si256((__m256i*)&grid[i][(j + L - 1) % L]);
            __m256i spin_neigh_s = _mm256_loadu_si256((__m256i*)&grid[(i + 1) % L][j]);

            __m256i delta_spin_neigh_n = _mm256_sub_epi32(spin_neigh_n, spin_old);
            __m256i delta_spin_neigh_e = _mm256_sub_epi32(spin_neigh_e, spin_old);
            __m256i delta_spin_neigh_w = _mm256_sub_epi32(spin_neigh_w, spin_old);
            __m256i delta_spin_neigh_s = _mm256_sub_epi32(spin_neigh_s, spin_old);

            // computing h_before and h_after
            __m256i h_before = _mm256_mul_epi32(spin_old, _mm256_add_epi32(_mm256_add_epi32(delta_spin_neigh_n, delta_spin_neigh_e),
                _mm256_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s)));
            __m256i h_after = _mm256_mul_epi32(spin_new, _mm256_add_epi32(_mm256_add_epi32(delta_spin_neigh_n, delta_spin_neigh_e),
                _mm256_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s)));

            __m256i delta_E = _mm256_sub_epi32(h_after, h_before);
            __m256 mask1 = _mm256_cmp_ps(_mm256_set1_ps(temp), _mm256_setzero_ps(), _CMP_GT_OQ);
            __m256 mask2 = _mm256_cmp_ps(_mm256_set1_ps(expf(-1 / temp)), _mm256_set1_ps(rand() / (float)RAND_MAX), _CMP_LE_OQ);

            __m256i accept_mask = _mm256_or_si256(_mm256_cmpgt_epi32(delta_E, _mm256_setzero_si256()), (__m256i)mask2);
            __m256i result = _mm256_blendv_epi32(spin_old, spin_new, accept_mask);

            _mm256_storeu_si256((__m256i*)&grid[i][j], result);
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
