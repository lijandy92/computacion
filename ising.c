#include "ising.h"

#include <math.h>
#include <stdlib.h>

//Modificada 
#include <xmmintrin.h> // Requiere la inclusión de este encabezado para utilizar tipos de datos vectoriales SSE.

void update(const float temp, int grid[L][L])
{
    const int VECTOR_WIDTH = 4; // Ancho del vector utilizado por el compilador. Puede variar según la arquitectura.

    // typewriter update
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += VECTOR_WIDTH) {
            // Cargar los valores de la matriz grid en un tipo de datos vectorial.
            __m128i spin_old_vec = _mm_loadu_si128((__m128i*)&grid[i][j]);

            // Calcular los valores de los vecinos utilizando operaciones vectoriales.
            __m128i spin_neigh_n_vec = _mm_loadu_si128((__m128i*)&grid[(i + L - 1) % L][j]);
            __m128i spin_neigh_e_vec = _mm_loadu_si128((__m128i*)&grid[i][(j + 1) % L]);
            __m128i spin_neigh_w_vec = _mm_loadu_si128((__m128i*)&grid[i][(j + L - 1) % L]);
            __m128i spin_neigh_s_vec = _mm_loadu_si128((__m128i*)&grid[(i + 1) % L][j]);

            // Realizar las operaciones vectoriales para calcular h_before y h_after.
            __m128i delta_spin_neigh_n_vec = _mm_sub_epi32(spin_neigh_n_vec, spin_old_vec);
            __m128i delta_spin_neigh_e_vec = _mm_sub_epi32(spin_neigh_e_vec, spin_old_vec);
            __m128i delta_spin_neigh_w_vec = _mm_sub_epi32(spin_neigh_w_vec, spin_old_vec);
            __m128i delta_spin_neigh_s_vec = _mm_sub_epi32(spin_neigh_s_vec, spin_old_vec);

            __m128i h_before_vec = _mm_mullo_epi32(spin_old_vec, _mm_add_epi32(_mm_add_epi32(delta_spin_neigh_n_vec, delta_spin_neigh_e_vec),
                                                                                _mm_add_epi32(delta_spin_neigh_w_vec, delta_spin_neigh_s_vec)));
            __m128i h_after_vec = _mm_mullo_epi32(_mm_set1_epi32(-1), _mm_add_epi32(_mm_add_epi32(delta_spin_neigh_n_vec, delta_spin_neigh_e_vec),
                                                                                   _mm_add_epi32(delta_spin_neigh_w_vec, delta_spin_neigh_s_vec)));

            // Comparar los resultados y almacenar el nuevo valor en la matriz grid.
            for (int k = 0; k < VECTOR_WIDTH; ++k) {
                int spin_old = _mm_extract_epi32(spin_old_vec, k);
                int h_before = _mm_extract_epi32(h_before_vec, k);
                int h_after = _mm_extract_epi32(h_after_vec, k);
                int delta_E = h_after - h_before;
                float p = rand() / (float)RAND_MAX;
                if (delta_E <= 0 || p <= expf(-delta_E / temp)) {
                    grid[i][j + k] = -spin_old;
                }
            }
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
