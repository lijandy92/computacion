#include "ising.h"

#include <math.h>
#include <stdlib.h>

//Modificada Comentada
void update(const float temp, int grid[L][L])
{
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; j += 8) { // Procesar de a 8 elementos a la vez (AVX)
            // Cargar elementos del grid en un registro AVX
            __m256i grid_values = _mm256_loadu_si256((__m256i*)&grid[i][j]);

            // Negar los valores del grid (cambiar de signo)
            __m256i negated_values = _mm256_sub_epi32(_mm256_setzero_si256(), grid_values);

            // Calcular los vecinos utilizando instrucciones intrínsecas AVX
            __m256i spin_neigh_n = _mm256_loadu_si256((__m256i*)&grid[(i + 1) % L][j]);
            __m256i spin_neigh_e = _mm256_loadu_si256((__m256i*)&grid[i][(j + 1) % L]);
            __m256i spin_neigh_w = _mm256_loadu_si256((__m256i*)&grid[i][(j + L - 1) % L]);
            __m256i spin_neigh_s = _mm256_loadu_si256((__m256i*)&grid[(i + L - 1) % L][j]);

            // Calcular las contribuciones de energía utilizando instrucciones intrínsecas AVX
            __m256i delta_spin_neigh_n = _mm256_sub_epi32(spin_neigh_n, grid_values);
            __m256i delta_spin_neigh_e = _mm256_sub_epi32(spin_neigh_e, grid_values);
            __m256i delta_spin_neigh_w = _mm256_sub_epi32(spin_neigh_w, grid_values);
            __m256i delta_spin_neigh_s = _mm256_sub_epi32(spin_neigh_s, grid_values);
            
            // Calcular h_before y h_after utilizando instrucciones intrínsecas AVX
            __m256i h_before = _mm256_mul_epi32(negated_values, _mm256_add_epi32(delta_spin_neigh_n, _mm256_add_epi32(delta_spin_neigh_e, _mm256_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s))));
            __m256i h_after = _mm256_mul_epi32(grid_values, _mm256_add_epi32(delta_spin_neigh_n, _mm256_add_epi32(delta_spin_neigh_e, _mm256_add_epi32(delta_spin_neigh_w, delta_spin_neigh_s))));

            // Calcular delta_E utilizando instrucciones intrínsecas AVX
            __m256i delta_E = _mm256_sub_epi32(h_after, h_before);

            // Generar una máscara de aceptación utilizando instrucciones intrínsecas AVX
            __m256 accept_mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), delta_E);

            // Generar un número aleatorio utilizando instrucciones intrínsecas AVX
            __m256 rand_values = _mm256_cvtepi32_ps(_mm256_set_epi32(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand()));
            __m256 rand_max = _mm256_set1_ps(RAND_MAX);
            __m256 p = _mm256_div_ps(rand_values, rand_max);

            // Calcular la condición de aceptación utilizando instrucciones intrínsecas AVX
            __m256 condition = _mm256_cmple_ps(p, _mm256_exp_ps(_mm256_div_ps(_mm256_cvtepi32_ps(delta_E), _mm256_set1_ps(temp))));

            // Aplicar la condición de aceptación utilizando instrucciones intrínsecas AVX
            __m256i result = _mm256_blendv_epi32(grid_values, negated_values, _mm256_castps_si256(condition));

            // Almacenar los resultados de vuelta en el grid
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
