#include "ising.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <emmintrin.h>
#include <random> // Agregado para incluir generador de números aleatorios
#include <cmath> // Agregado para incluir función exponencial

void update(const float temp, int8_t grid[L][L]) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            int8_t spin = grid[i][j];
            int8_t spin_neigh_e = grid[i][(j + 1) % L];
            int8_t spin_neigh_w = grid[i][(j + L - 1) % L];
            int8_t spin_neigh_s = grid[(i + 1) % L][j];
            int8_t spin_neigh_n = grid[(i + L - 1) % L][j];

            int8_t delta_spin_neigh_n = spin_neigh_n - spin;
            int8_t delta_spin_neigh_e = spin_neigh_e - spin;
            int8_t delta_spin_neigh_w = spin_neigh_w - spin;
            int8_t delta_spin_neigh_s = spin_neigh_s - spin;

            int8_t h_before = -spin * (delta_spin_neigh_n + delta_spin_neigh_e + delta_spin_neigh_w + delta_spin_neigh_s);
            int8_t h_after = -spin * (delta_spin_neigh_n + delta_spin_neigh_e + delta_spin_neigh_w + delta_spin_neigh_s);

            int8_t delta_E = h_after - h_before;
            float p = dis(gen);

            if (delta_E <= 0 || p <= expf(-delta_E / temp)) {
                grid[i][j] = -spin;
            }
        }
    }
}

double calculate(int8_t grid[L][L], int* M_max) {
    int E = 0;

    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            int8_t spin = grid[i][j];
            int8_t spin_neigh_e = grid[i][(j + 1) % L];
            int8_t spin_neigh_w = grid[i][(j + L - 1) % L];
            int8_t spin_neigh_s = grid[(i + 1) % L][j];
            int8_t spin_neigh_n = grid[(i + L - 1) % L][j];

            E -= spin * (spin_neigh_n + spin_neigh_e + spin_neigh_w + spin_neigh_s);
            *M_max += spin;
        }
    }

    return -((double)E / 2.0);
}
