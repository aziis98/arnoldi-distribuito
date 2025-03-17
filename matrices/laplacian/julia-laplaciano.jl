using SparseArrays
using MAT

N = parse(Int, ARGS[1])
println("Generating 3D Laplacian for size $N")

nx = N
ny = N
nz = N

ex = fill(1, nx)
ey = fill(1, ny)
ez = fill(1, nz)

Dxx = spdiagm(-1 => ex, 0 => -2 * ex, +1 => ex)
Dyy = spdiagm(-1 => ey, 0 => -2 * ey, +1 => ey)
Dzz = spdiagm(-1 => ez, 0 => -2 * ez, +1 => ez)

Ix = spdiagm(0 => [ex; 1])
Iy = spdiagm(0 => [ey; 1])
Iz = spdiagm(0 => [ez; 1])

L = kron(Dxx, Iy, Iz) + kron(Ix, Dyy, Iz) + kron(Ix, Iy, Dzz)

display(sparse(L))

matwrite("laplacian_$N.mat", Dict("A" => L))
