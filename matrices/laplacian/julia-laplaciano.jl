using SparseArrays
using MAT

# 11 x 16
nx = 10
ny = 15
ex = fill(1, nx)
ey = fill(1, ny)
Dxx = spdiagm(-1 => ex, 0 => -2 * ex, +1 => ex)
Dyy = spdiagm(-1 => ey, 0 => -2 * ey, +1 => ey)
L = kron(Dyy, spdiagm(0 => [ex; 1])) + kron(spdiagm(0 => [ey; 1]), Dxx);

matwrite("laplacian-discretization-3d.mat", Dict("A" => L))
