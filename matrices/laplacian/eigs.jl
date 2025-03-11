using SparseArrays
using LinearAlgebra
using MAT

# 11 x 16
nx = 10
ny = 15
ex = fill(1, nx)
ey = fill(1, ny)
Dxx = diagm(-1 => ex, 0 => -2 * ex, +1 => ex)
Dyy = diagm(-1 => ey, 0 => -2 * ey, +1 => ey)
L = kron(Dyy, diagm(0 => [ex; 1])) + kron(diagm(0 => [ey; 1]), Dxx);

println("Laplacian matrix L:")

display(sparse(L))
display(eigvals(L))