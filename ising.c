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
            int spin_old[4] = { grid[i][j], grid[i][j + 1], grid[i][j + 2], grid[i][j + 3] };
            int spin_new[4] = { -spin_old[0], -spin_old[1], -spin_old[2], -spin_old[3] };

            // computing energy contributions of neighbors
            int spin_neigh_n[4] = { grid[(i + L - 1) % L][j], grid[(i + L - 1) % L][j + 1],
                                    grid[(i + L - 1) % L][j + 2], grid[(i + L - 1) % L][j + 3] };
            int spin_neigh_e[4] = { grid[i][(j + 1) % L], grid[i][(j + 2) % L],
                                    grid[i][(j + 3) % L], grid[i][(j + 4) % L] };
            int spin_neigh_w[4] = { grid[i][(j + L - 1) % L], grid[i][(j + L - 2) % L],
                                    grid[i][(j + L - 3) % L], grid[i][(j + L - 4) % L] };
            int spin_neigh_s[4] = { grid[(i + 1) % L][j], grid[(i + 1) % L][j + 1],
                                    grid[(i + 1) % L][j + 2], grid[(i + 1) % L][j + 3] };
            
            int delta_spin_neigh_n[4] = { spin_neigh_n[0] - spin_old[0], spin_neigh_n[1] - spin_old[1],
                                          spin_neigh_n[2] - spin_old[2], spin_neigh_n[3] - spin_old[3] };
            int delta_spin_neigh_e[4] = { spin_neigh_e[0] - spin_old[0], spin_neigh_e[1] - spin_old[1],
                                          spin_neigh_e[2] - spin_old[2], spin_neigh_e[3] - spin_old[3] };
            int delta_spin_neigh_w[4] = { spin_neigh_w[0] - spin_old[0], spin_neigh_w[1] - spin_old[1],
                                          spin_neigh_w[2] - spin_old[2], spin_neigh_w[3] - spin_old[3] };
            int delta_spin_neigh_s[4] = { spin_neigh_s[0] - spin_old[0], spin_neigh_s[1] - spin_old[1],
                                          spin_neigh_s[2] - spin_old[2], spin_neigh_s[3] - spin_old[3] };
            
            // computing h_before and h_after
            int h_before[4] = { -spin_old[0] * (delta_spin_neigh_n[0] + delta_spin_neigh_e[0] + delta_spin_neigh_w[0] + delta_spin_neigh_s[0]),
                                -spin_old[1] * (delta_spin_neigh_n[1] + delta_spin_neigh_e[1] + delta_spin_neigh_w[1] + delta_spin_neigh_s[1]),
                                -spin_old[2] * (delta_spin_neigh_n[2] + delta_spin_neigh_e[2] + delta_spin_neigh_w[2] + delta_spin_neigh_s[2]),
                                -spin_old[3] * (delta_spin_neigh_n[3] + delta_spin_neigh_e[3] + delta_spin_neigh_w[3] + delta_spin_neigh_s[3]) };
            int h_after[4] = { -spin_new[0] * (delta_spin_neigh_n[0] + delta_spin_neigh_e[0] + delta_spin_neigh_w[0] + delta_spin_neigh_s[0]),
                               -spin_new[1] * (delta_spin_neigh_n[1] + delta_spin_neigh_e[1] + delta_spin_neigh_w[1] + delta_spin_neigh_s[1]),
                               -spin_new[2] * (delta_spin_neigh_n[2] + delta_spin_neigh_e[2] + delta_spin_neigh_w[2] + delta_spin_neigh_s[2]),
                               -spin_new[3] * (delta_spin_neigh_n[3] + delta_spin_neigh_e[3] + delta_spin_neigh_w[3] + delta_spin_neigh_s[3]) };

            int delta_E[4] = { h_after[0] - h_before[0], h_after[1] - h_before[1],
                               h_after[2] - h_before[2], h_after[3] - h_before[3] };

            // Generate random values
            float rand_vals[4] = { rand() / (float)RAND_MAX, rand() / (float)RAND_MAX,
                                   rand() / (float)RAND_MAX, rand() / (float)RAND_MAX };

            // Compare delta_E with zero and p with exp(-delta_E / temp)
            __m128 accept_mask = _mm_or_ps(_mm_cmple_ps(_mm_setzero_ps(), _mm_cvtepi32_ps(delta_E)),
                                           _mm_cmple_ps(_mm_set_ps(rand_vals[3], rand_vals[2], rand_vals[1], rand_vals[0]),
                                                        _mm_div_ps(_mm_cvtepi32_ps(delta_E), _mm_set1_ps(temp))));

            // Blend spin_old and spin_new based on the accept_mask
            __m128i result = _mm_blendv_epi32(_mm_loadu_si128((__m128i*)spin_old),
                                               _mm_loadu_si128((__m128i*)spin_new), accept_mask);

            // Store the updated spins back to the grid
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
