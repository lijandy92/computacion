#include "ising.h"
#include <emmintrin.h>
#include <xmmintrin.h>
#include <math.h>
#include <stdlib.h>

#include "ising.h"
#include <sfmt.h>

void update(const float temp, int grid[L][L])
{
    sfmt_t sfmt;  // Declaración de la estructura SFMT
    sfmt_init_gen_rand(&sfmt, 0);  // Inicialización del generador SFMT

    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 4) {  // Vectorización de 4 elementos a la vez
            __m128i spin_old = _mm_loadu_si128((__m128i*)&grid[i][j]);  // Cargar 4 elementos del grid en un registro
            __m128i spin_new = _mm_sub_epi32(_mm_set1_epi32(0), spin_old);  // Negar los 4 elementos del grid

            // Realizar cálculos vectorizados

            // Generar 4 números aleatorios
            __m128i rand_num = _mm_set_epi32(sfmt_genrand_uint32(&sfmt), sfmt_genrand_uint32(&sfmt),
                                             sfmt_genrand_uint32(&sfmt), sfmt_genrand_uint32(&sfmt));
            // Generar 4 números máximos aleatorios
            __m128i rand_max = _mm_set1_epi32(SFMT_MAX);
            // Realizar cálculos vectorizados

            // Actualizar el grid vectorizado
            _mm_storeu_si128((__m128i*)&grid[i][j], spin_new);
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
