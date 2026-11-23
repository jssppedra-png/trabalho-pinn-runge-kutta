| Problema | Método | Erro máximo | Tempo (s) | Avaliações de f |
|---|---|---|---|---|
| (i) y' = -50y | RK45, tolerância 1e-10 | 9,39e-11 | 1,09e-02 | 884 |
| (i) y' = -50y | PINN, 5 mil iterações | 9,95e-01 | 1,80e+01 | 1.000.000 |
| (i) y' = -50y | PINN, fiscal do início x 100 | 6,22e-03 | 1,60e+01 | 1.000.000 |
| (i) y' = -50y | PINN, curva amarrada no início | 1,89e-01 | 1,69e+01 | 1.000.000 |
| (ii) y' = cos(20t) | RK45, tolerância 1e-10 | 6,46e-10 | 8,30e-03 | 740 |
| (ii) y' = cos(20t) | PINN, 5 mil iterações | 3,91e-04 | 1,73e+01 | 1.000.000 |
| (iii) coelhos e raposas | RK45, tolerância 1e-10 | 1,84e-09 | 2,75e-02 | 1.256 |
| (iii) coelhos e raposas | PINN, 5 mil iterações | 9,63e-01 | 2,35e+01 | 1.000.000 |
| (iii) coelhos e raposas | PINN, 20 mil iterações | 5,59e-03 | 9,03e+01 | 4.000.000 |
