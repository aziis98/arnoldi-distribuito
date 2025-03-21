using SparseArrays
using MAT

N = parse(Int, ARGS[1])
println("Generating 3D Laplacian for size $N")

nx = N
ny = N
nz = N

ex = fill(1, nx - 1)
ey = fill(1, ny - 1)
ez = fill(1, nz - 1)

Dxx = spdiagm(-1 => ex, 0 => -2 * [ex; 1], +1 => ex)
Dyy = spdiagm(-1 => ey, 0 => -2 * [ey; 1], +1 => ey)
Dzz = spdiagm(-1 => ez, 0 => -2 * [ez; 1], +1 => ez)

Ix = spdiagm(0 => ex)
Iy = spdiagm(0 => ey)
Iz = spdiagm(0 => ez)

L = kron(Dxx, Iy, Iz) + kron(Ix, Dyy, Iz) + kron(Ix, Iy, Dzz)

display(sparse(L))

matwrite("laplacian_$N.mat", Dict("A" => L))
