

//Modificada Comentada
#include "ising.h"
#include <math.h>
#include <stdlib.h>
#include <xmmintrin.h>

#include "ising.h"
#include <math.h>
#include <stdlib.h>
#include <emmintrin.h>

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

            // Convert mask_delta_E to a mask for maskload
            int mask = _mm_movemask_ps(_mm_castsi128_ps(mask_delta_E));

            // Mask load spin_old using the mask
            __m128i masked_spin_old = _mm_maskload_epi32((const int*)&grid[i][j], mask);

            // Mask spin_new using the mask
            __m128i masked_spin_new = _mm_and_si128(spin_new, mask_delta_E);

            // Merge masked_spin_old and masked_spin_new
            __m128i updated_spin = _mm_or_si128(masked_spin_old, masked_spin_new);

            // Mask store updated_spin back to grid using the mask
            _mm_maskstore_epi32((int*)&grid[i][j], mask, updated_spin);
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
