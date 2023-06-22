#include "ising.h"
#include <math.h>
#include <stdlib.h>
#include <emmintrin.h>

void update(const float temp, int grid[L][L])
{
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 4) {
            // Cargar 4 elementos consecutivos de la matriz en un registro SIMD
            __m128i spin_old = _mm_loadu_si128((__m128i*)&grid[i][j]);
            
            // Calcular posiciones de vecinos
            unsigned int i_neigh_n = (i + L - 1) % L;
            unsigned int i_neigh_e = i;
            unsigned int i_neigh_w = i;
            unsigned int i_neigh_s = (i + 1) % L;
            
            // Cargar 4 elementos consecutivos de los vecinos en un registro SIMD
            __m128i spin_neigh_n = _mm_loadu_si128((__m128i*)&grid[i_neigh_n][j]);
            __m128i spin_neigh_e = _mm_loadu_si128((__m128i*)&grid[i_neigh_e][(j + 1) % L]);
            __m128i spin_neigh_w = _mm_loadu_si128((__m128i*)&grid[i_neigh_w][(j + L - 1) % L]);
            __m128i spin_neigh_s = _mm_loadu_si128((__m128i*)&grid[i_neigh_s][j]);
            
            // Calcular diferencias de espines entre vecinos y espín antiguo
            __m128i delta_spin_neigh_n = _mm_sub_epi32(spin_neigh_n, spin_old);
            __m128i delta_spin_neigh_e = _mm_sub_epi32(spin_neigh_e, spin_old);
            __m128i delta_spin_neigh_w = _mm_sub_epi32(spin_neigh_w, spin_old);
            __m128i delta_spin_neigh_s = _mm_sub_epi32(spin_neigh_s, spin_old);
            
            // Calcular h_before y h_after utilizando operaciones vectoriales
            __m128i h_before = _mm_mullo_epi32(spin_old, _mm_add_epi32(delta_spin_neigh_n,
                                                                        _mm_add_epi32(delta_spin_neigh_e,
                                                                                      _mm_add_epi32(delta_spin_neigh_w,
                                                                                                    delta_spin_neigh_s))));
            __m128i h_after = _mm_mullo_epi32(_mm_set1_epi32(-1), h_before);
            
            // Calcular delta_E utilizando operaciones vectoriales
            __m128i delta_E = _mm_sub_epi32(h_after, h_before);
            
            // Calcular p utilizando operaciones vectoriales
            __m128 p = _mm_cvtepi32_ps(_mm_div_epi32(_mm_set1_epi32(rand()), _mm_set1_epi32(RAND_MAX)));
            
            // Calcular máscara de actualización utilizando operaciones vectoriales
            __m128 mask = _mm_cmple_ps(delta_E, _mm_setzero_ps());
            
            // Calcular espín actualizado utilizando operaciones vectoriales
            __m128i spin_new = _mm_or_si128(_mm_and_si128((__m128i)mask, (__m128i)spin_new), _mm_andnot_si128((__m128i)mask, (__m128i)spin_old));
            
            // Almacenar espín actualizado en la matriz
            _mm_storeu_si128((__m128i*)&grid[i][j], spin_new);
        }
    }
}

double calculate(int grid[L][L], int* M_max)
{
    int E = 0;
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 4) {
            // Cargar 4 elementos consecutivos de la matriz en un registro SIMD
            __m128i spin = _mm_loadu_si128((__m128i*)&grid[i][j]);
            
            // Calcular posiciones de vecinos
            unsigned int i_neigh_n = (i + 1) % L;
            unsigned int i_neigh_e = i;
            unsigned int i_neigh_w = i;
            unsigned int i_neigh_s = (i + L - 1) % L;
            
            // Cargar 4 elementos consecutivos de los vecinos en un registro SIMD
            __m128i spin_neigh_n = _mm_loadu_si128((__m128i*)&grid[i_neigh_n][j]);
            __m128i spin_neigh_e = _mm_loadu_si128((__m128i*)&grid[i_neigh_e][(j + 1) % L]);
            __m128i spin_neigh_w = _mm_loadu_si128((__m128i*)&grid[i_neigh_w][(j + L - 1) % L]);
            __m128i spin_neigh_s = _mm_loadu_si128((__m128i*)&grid[i_neigh_s][j]);
            
            // Calcular contribuciones de energía utilizando operaciones vectoriales
            __m128i energy_contributions = _mm_add_epi32(_mm_mullo_epi32(spin, spin_neigh_n),
                                                        _mm_add_epi32(_mm_mullo_epi32(spin, spin_neigh_e),
                                                                      _mm_add_epi32(_mm_mullo_epi32(spin, spin_neigh_w),
                                                                                    _mm_mullo_epi32(spin, spin_neigh_s))));
            
            // Sumar contribuciones de energía utilizando operaciones vectoriales
            __m128i energy_sum = _mm_hadd_epi32(energy_contributions, energy_contributions);
            energy_sum = _mm_hadd_epi32(energy_sum, energy_sum);
            
            // Actualizar valor de E
            E += _mm_cvtsi128_si32(energy_sum) / 2;
            
            // Actualizar valor de M_max
            *M_max += _mm_popcnt_u32(_mm_movemask_epi8(spin));
        }
    }
    
    return -((double)E / 2.0);
}