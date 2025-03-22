using SparseArrays
using LinearAlgebra
using MAT

# 20 x 20 x 20 grid
nx = 20 - 1
ny = 20 - 1
nz = 20 - 1

ex = fill(1, nx)
ey = fill(1, ny)
ez = fill(1, nz)

Dxx = diagm(-1 => ex, 0 => -2 * ex, +1 => ex)
Dyy = diagm(-1 => ey, 0 => -2 * ey, +1 => ey)
Dzz = diagm(-1 => ez, 0 => -2 * ez, +1 => ez)

Ix = diagm(0 => [ex; 1])
Iy = diagm(0 => [ey; 1])
Iz = diagm(0 => [ez; 1])

L = kron(Dxx, Iy, Iz) + kron(Ix, Dyy, Iz) + kron(Ix, Iy, Dzz)

# display(eigvals(L))

# # 10 x 17 grid
# nx = 11 - 1
# ny = 16 - 1

# ex = fill(1, nx)
# ey = fill(1, ny)

# Dxx = spdiagm(-1 => ex, 0 => -2 * ex, +1 => ex)
# Dyy = spdiagm(-1 => ey, 0 => -2 * ey, +1 => ey)

# Ix = spdiagm(0 => [ex; 1])
# Iy = spdiagm(0 => [ey; 1])

# L = kron(Dxx, Iy) + kron(Ix, Dyy)

println("Laplacian matrix L:")

display(sparse(L))

l = 400

# arnoldi iteration
Q = Matrix{Float64}(undef, size(L, 1), l)
Q[:, 1] = ones(size(L, 1))
H = zeros(l, l)

for j in 2:l
    # Arnoldi iteration
    v = L * Q[:, j - 1]
    
    # Reorthogonalization
    for i in 1:(j - 1)
        H[i, j - 1] = dot(Q[:, i], v)
        v -= H[i, j - 1] * Q[:, i]
    end
    H[j, j - 1] = norm(v)
    if H[j, j - 1] < 1e-10
        break
    end
    Q[:, j] = v / H[j, j - 1]
end

H = H[1:(l-1), 1:(l-1)]

println("Hessenberg matrix H:")
display(sparse(H))

display(eigvals(H))
