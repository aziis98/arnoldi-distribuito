#import "theme.typ": ams-article, definition, theorem, proposition, proof, todo

#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#show: ams-article.with(
  paper-size: "a4",
  title: [Distributed Arnoldi Method for Eigenvalues using PETSc & Lapack],
  authors: (
    (
      name: "Antonio De Lucreziis",
      organization: [Dipartimento di Matematica],
      location: [Pisa, Italia],
      email: "antonio.delucreziis@gmail.com",
      url: "https://poisson.phc.dm.unipi.it/~delucreziis/"
    ),
  ),
  abstract: [
    In this project we implement the Arnoldi Method for eigenvalues using #link("https://www.netlib.org/lapack/")[Lapack], #link("https://petsc.org/release/")[PETSc], #link("https://www.open-mpi.org/")[OpenMPI] libraries for distributed matrix and vector operations and run some numerical simulations on a small cluster of $tilde.basic 30$ nodes of #link("https://radxa.com/products/rock4/4cp")[Rock 4C+] (`aarch64`) boards we built for the course of #link("https://fdurastante.github.io/courses/highperfmathematics24.html")[High-Performance Mathematics]. The code can be found at #link("https://github.com/aziis98/arnoldi-distribuito")[github.com/aziis98/arnoldi-distribuito].
  ]
)

#outline()

= Arnoldi Method for Eigenvalues

== Theory of Krylov subspaces

#definition[
  The Krylov subspace of order $ell$ associated with $A$ and $b$ is the subspace
  $
    cal(K)_ell (A, b) colon.eq op("span")(b, A b, dots, A^(ell-1) b)
  $
]

#definition[
  If $x_ell$ is an approximation of the solution to $A x = b$ then the residual is defined as
  $
    r_ell colon.eq b - A x_ell
  $
]

There are various criteria that can be used to identify the "optimal" solution

- *FOM* -- Have the residual be orthogonal to $cal(K)_ell(A, b)$

- *GMRES* -- Have the residual minimized such that $x_ell$ is
  $
    x_ell colon.eq op("argmin")_(x in cal(K)_ell(A, b)) norm(b - A x)_2
  $

We are now interested in studying the *FOM* case and its behavior when $ell << n$.

== Arnoldi Iteration

We now want to construct a nice basis for $cal(K)_ell(A, b)$, as the one directly given by $A^j b$ has very poor computational properties. We need a way of constructing an orthogonal basis of the Krylov subspace without using the QR algorithm.

Let's define the Arnoldi iteration that works as follows:

- First let's define $v_1 colon.eq b slash norm(b)_2$

- For $j = 1, dots, ell$

  - Let's define $w_(j+1) colon.eq A v_j$
  
  - The to obtain the next vector let's orthogonalize it to the current basis $v_1, dots,  v_j$ and let
  
    $
    pi_j(w_(j+1)) colon.eq sum_(i <= j) (v_i^* w_(j+1)) v_i \
    v_(j+1) 
    colon.eq
    (w_(j+1) - pi_j(w_(j+1)))/(norm(w_(j+1) - pi_j(w_(j+1)))_2)
    $

Let's also define the following *upper Hessenberg matrix* composed of all the projection coefficients

$
  h_(i,j) colon.eq v_i^* w_(j+1) = v_i^* A v_j \
  H_ell colon.eq
  mat(
    h_(1,1), dots, dots, h_(1,ell);
    h_(2,1), h_(2,2), "", dots.v;
    "", dots.down, dots.down, dots.v;
    "", "", h_(ell,ell-1), h_(ell,ell);
  )
$

The final algorithm for the *Arnoldi iteration* is

#align(center, 
  block(
    width: 18em,
    align(left, 
      algorithm({
        import algorithmic: *
        Function("Arnoldi-Iteration", args: ($A$, $b$, $ell$), {
          Assign[$q_1$][$b \/ norm(b)_2$]
          For(cond: $k = 2, .., ell$, {
            Assign[$q_k$][$A q_(k-1)$]
            For(cond: $j = 1, ..., k - 1$, {
              Assign[$h_(j,k-1)$][$q_j^* q_k$]
              Assign[$q_k$][$q_k - h_(j,k-1) q_j$]
            })
            Assign[$h_(k,k-1)$][$norm(q_k)_2$]
            Assign[$q_k$][$q_k \/ h_(k,k-1)$]
          })
        })
      })
    )
  )
)

== Arnoldi iteration for eigenvalues and eigenvectors

To use the Arnoldi iteration to compute the eigenvalues of $A$ we can use the following procedure:

- First we use the Arnoldi iteration to compute the upper Hessenberg matrix $H_ell$ for the Krylov subspace $cal(K)_ell(A, b)$.

- We then compute the eigenvalues and eigenvectors of this Hessenberg matrix by computing a QR decomposition of the matrix $H_ell$.

The following theorem states the relationship between the eigenvalues of $H_ell$ and $A$

#theorem[
  Let $H_ell$ be the upper Hessenberg matrix from the Arnoldi process for $A$ (without breakdown), starting from $b$. Then, the characteristic polynomial $hat(p)(lambda) = det(lambda I - H_ell)$ satisfies
  $
    hat(p) = op("argmin", limits: #true)_(p in cal(P)_ell \ p_ell = 1) norm(p(A) b)_2
  $
]

So the characteristic polynomial of $H_ell$ is a good approximation of the characteristic polynomial of $A$.

#v(1em)

To benchmark the performance of our implementation of the distributed Arnoldi method we will use the test problem of finding the eigenvalues of the 3D Laplacian that has a nice closed form solution. 

== 3D Laplacian

The matrix we will consider comes up from the Laplace equation. Let $u : RR^n -> RR$, for example we can consider the zero Dirichlet boundary condition on the unit cube $[0, 1]^n$ and get the following problem

$
cases(
  laplace u(x) = 0 #h(4em) & x in [0, 1]^3,
  u(x) = u_0(x) & x in partial [0, 1]^3
)
$

We now discretize this problem using the finite difference method. We will use a uniform grid with $n_i$ points in the $i$-th direction (and also let $h_i colon.eq 1 slash (n_i + 1)$). In dimension $n = 1$ the problem is just the second derivative of $u$ with respect to $x$ and the finite difference approximation is

$
  D_(x x) u(x) = (u(x + h_x) - 2 u(x) + u(x - h_x)) / (h_x^2)
$

where $h_x colon.eq 1 slash (n_1 + 1)$ is the distance between two consecutive points. Notice that here the stencil has shape $mat(1, -2, 1, delim: "[")$. If we consider the matrix $upright(bold(D))_(x x)$ associated with the finite difference operator, we can write the problem as $upright(bold(D))_(x x) overline(u) = 0$ where $overline(u)$ is the vector of values of $u$ at the grid points, and $upright(bold(D))_(x x)$ is the following tridiagonal matrix

$
  upright(bold(D))_(x x) =
  1 / h_1^2
  mat(
    -2, 1,  , , ;
    1, -2, 1, , ;
    , dots.down, dots.down, dots.down, ;
    , , 1, -2, 1;
    , , , 1, -2;
    delim: "["
  )
$

It is well known that a tridiagonal matrix with constant diagonals of $a$, $b$, $c$ has eigenvalues
$
  lambda_k = a - 2 sqrt(b c) cos((pi k) / (n + 1))
$
for $k = 1, dots, n$. In this case we have $a = -2$, $b = 1$, $c = 1$ so the eigenvalues are given by the following expression (and have the following bounds):
$
  lambda_k 
  &= -2 - 2 cos((pi k) / (n + 1)) \
  & = -2 (1 + cos(pi k / (n + 1)), in [-1, 1]) \
  &= -4 underbrace(cos^2(pi / 2 dot k / (n + 1)), in [0, 1])
  ==> -4 < lambda_k < 0 \
$

For dimension $n = 2$ we have the following stencils:

$
  mat(
    , 1,  ;
    1, -4, 1;
    , 1,  ;
    delim: "["
  )
  =
  mat(1, -2, 1; delim: "[")
  +
  mat(1;-2;1; delim: "[")
  \
  #rotate(90deg, $ arrow.squiggly $)
  \
  (D_(x x) + D_(y y)) u(x, y) = (u(x - h_1, y) + u(x + h_1, y) + u(x, y - h_y) + u(x, y + h_y) - 4 u(x, y)) / (h_1^2)
$

In this case the matrix associated with the finite difference operator can be written compactly as the sum of two Kronecker products:
$
  D_(x x) + D_(y y) =
  upright(bold(I)) times.circle upright(bold(D))_(y y)
  +
  upright(bold(D))_(x x) times.circle upright(bold(I))
$

where $upright(bold(I))$ is the identity matrix.

This can also be generalized to the case of $n = 3$ and higher dimensions. The matrix for the discretized 3D Laplacian can be written as

$
  L = upright(bold(D_(x x))) times.circle upright(bold(I)) times.circle upright(bold(I))
    + upright(bold(I)) times.circle upright(bold(D_(y y))) times.circle upright(bold(I))
    + upright(bold(I)) times.circle upright(bold(I)) times.circle upright(bold(D_(z z)))
$

We are now interested in finding the eigenvalues of this matrix for large $N$, we will drop the factor $1 slash h_i^2$ from the matrix and directly compute the eigenvalues of the matrix $L$.

To compute the eigenvalues and eigenvectors of this matrix we can use the following facts about the Kronecker products.

#proposition[
  The Kronecker product has the following properties

  - The Kronecker product distributes over the matrix product:
    $
      (A times.circle B) (C times.circle D) = (A C) times.circle (B D)
    $

  - Conjugation commutes with the Kronecker product, i.e.
    $
      (A times.circle B)^* = A^* times.circle B^*
    $

  - Let $A = A_1 times.circle A_2 times.circle dots times.circle A_n$ be the Kronecker product of $n$ matrices. Then, the eigenvalues of $A$ are given by the products of the eigenvalues of the $A_i$.
]

#proposition[
  Given two diagonalizable matrices $A$ and $B$ with eigenvalues $lambda_i$ and $mu_j$, respectively the expression
  $
    A times.circle I + I times.circle B
  $
  has eigenvalues $lambda_i + mu_j$ and a matrix of eigenvectors is $V times.circle W$ where $V$ and $W$ are the matrices of eigenvectors of $A$ and $B$, respectively
  $
    #grid(
      columns: 2,
      rows: 2,
      align: center + horizon,
      row-gutter: 1em,
      column-gutter: 2em,
      $
        A = V D_A V^*
      $,
      $
        B = W D_B W^*
      $,
      $
        D_A = mat(
          dots.down, , ;
          , lambda_i, ;
          , , dots.down;
          delim: "["
        )
      $,
      $
        D_B = mat(
          dots.down, , ;
          , mu_j, ;
          , , dots.down;
          delim: "["
        )
      $
    )
  $
]

#proof[
  Let's consider $D_A = V^* D_A V$ and $D_B = W^* D_B W$. Then, by conjugation with $V times.circle W$ we have
  $
    & (V times.circle W)^* (A times.circle I + I times.circle B) (V times.circle W) \
    &= (V^* times.circle W^*) (A times.circle I + I times.circle B) (V times.circle W) \
    &= (V^* times.circle W^*) (A times.circle I) (V times.circle W) + (V^* times.circle W^*) (I times.circle B) (V times.circle W) \
    &= (V^* A V times.circle I) + (I times.circle W^* B W) \
    &= D_A times.circle I + I times.circle D_B
  $
  To conclude let's analyze the block structure of this last term
  $
    D_A times.circle I + I times.circle D_B 
    = mat(
      dots.down, , ;
      , lambda_i #rect(stroke: 0.5pt, $I$), ;
      , , dots.down;
      delim: "["
    )
    +
    mat(
      delim: "[",
      dots.down, , ;
      , #rect(stroke: 0.5pt, $D_B$), ;
      , , dots.down;
    )
    = mat(
      dots.down, , ;
      , #rect(stroke: 0.5pt, $lambda_i I + D_B$), ;
      , , dots.down;
      delim: "["
    )
  $
  We can thus produce all possible combinations of sums of $lambda_i$ and $mu_j$ and so the final eigenvalues are given by
  $
    = mat(
      dots.down, , ;
      , lambda_i + mu_j, ;
      , , dots.down;
      delim: "["
    )
  $ 
]

This fact can also be generalized to the case of $n$ dimensions. Recalling that the eigenvalues of the 1D Laplacian are given by
$
  lambda_i = -4 cos^2((pi i) / (2(n + 1)))
$
we can write the eigenvalues of the 3D Laplacian as
$
  lambda_(i_1, i_2, i_3) =
  -4 (cos^2((pi i_1) / (2(n_1 + 1))) +
  cos^2((pi i_2) / (2(n_2 + 1))) + 
  cos^2((pi i_3) / (2(n_3 + 1))))

$
for $(i_1, i_2, i_3) in [1, dots, n_1] times [1, dots, n_2] times [1, dots, n_3]$. This is approximately in the range $[-12, 0]$.

// TODO

// The eigenvector for the 1D Laplacian is given by
// $
//   v_(i; thin j) =
//   sqrt(2 / (n + 1)) sin((pi i j) / (n + 1))
// $
// for $i = 1, dots, n_1$ so the eigenvector for the 3D Laplacian is
// $
//   v_(i_1, i_2, i_3; thin j_1, j_2, j_3) =
//   sqrt(2 / (n_1 + 1)) sin((pi i_1 j_1) / (n_1 + 1))
//   sqrt(2 / (n_2 + 1)) sin((pi i_2 j_2) / (n_2 + 1))
//   sqrt(2 / (n_3 + 1)) sin((pi i_3 j_3) / (n_3 + 1))
// $
// for the eigenvalue with multi-index $(i_1, i_2, i_3)$ and where $(j_1, j_2, j_3)$ is the multi-index for the various components of the eigenvector.

#pagebreak()

= PETSc Implementation

Let's look again at the algorithm for the Arnoldi iteration. There are various operations that can be done in parallel.

#align(center, 
  block(
    width: 18em,
    align(left, 
      algorithm({
        import algorithmic: *
        Function("Arnoldi-Iteration", args: ($A$, $b$, $ell$), {
          Assign[$q_1$][$b \/ norm(b)_2$]
          For(cond: $k = 2, .., ell$, {
            Assign[$q_k$][$A q_(k-1)$]
            For(cond: $j = 1, ..., k - 1$, {
              Assign[$h_(j,k-1)$][$q_j^* q_k$]
              Assign[$q_k$][$q_k - h_(j,k-1) q_j$]
            })
            Assign[$h_(k,k-1)$][$norm(q_k)_2$]
            Assign[$q_k$][$q_k \/ h_(k,k-1)$]
          })
        })
      })
    )
  )
)

If we store the matrix $A$ and the vector $b$ in a distributed way, we can compute the following operations in parallel:

- The normalization of $b$ using the function #link("https://petsc.org/release/manualpages/Vec/VecNormalize/")[`VecNormalize`].

- The matrix-vector product $A q_k$ using #link("https://petsc.org/release/manualpages/Mat/MatMult/")[`MatMult`].

- The inner product $q_j^* q_k$ using #link("https://petsc.org/release/manualpages/Vec/VecDot/")[`VecDot`].

- The operation $q_k - h_(j,k-1) q_j$ can be done in parallel as well. More precisely, this can be done as a _fused multiply-add_ operation using the PETSc #link("https://petsc.org/release/manualpages/Vec/VecAXPY/")[`VecAXPY`] function.

- The norm of $q_k$ can be computed using #link("https://petsc.org/release/manualpages/Vec/VecNorm/")[`VecNorm`].

The final algorithm written in C is the following: we will omit all calls to `PetscCall(...)` and other PETSc related infrastructure for the sake of brevity.

#align(center, box(fill: luma(93%), radius: 5pt, inset: 5pt)[
  #set text(size: 8.5pt)
  #set par(leading: 4pt)
  ```c
  PetscErrorCode ArnoldiIteration(Mat A, Vec b, PetscInt n, PetscInt m, Vec *Q, double *h) {
      Vec q;
      VecDuplicate(b, &q); 
      VecCopy(b, q); 
      VecNormalize(q, NULL);
      Q[0] = q;
      for (PetscInt k = 1; k < n + 1; k++) {
          Vec v;
          VecDuplicate(b, &v);
          MatMult(A, Q[k - 1], v);
          PetscScalar h_ij;
          for (PetscInt j = 0; j < k; j++) {
              VecDot(Q[j], v, &h_ij);
              h[j * (n + 1) + k - 1] = h_ij;
              VecAXPY(v, -h_ij, Q[j]);
          }
          VecNorm(v, NORM_2, &h_ij);
          h[j * (n + 1) + k - 1] = h_ij;
          if (h_ij > eps) {
              VecNormalize(v, NULL);
              Q[k] = v;
          } else {
              break; // Early breakdown
          }
      }
  }
  ```
])

The matrix $A$ is stored in a distributed way using the PETSc #link("https://petsc.org/release/manualpages/Mat/Mat/")[`Mat`] type. The vector $b$ is also stored in a distributed way using the PETSc #link("https://petsc.org/release/manualpages/Vec/Vec/")[`Vec`] type. In the first experiments the matrix $A$ is a 3D Laplacian matrix (symmetric case) and the vector $b$ is the vector of ones.

// #pagebreak()

= Numerical Experiments

For the numerical experiments we will use a Krylov subspace of dimension $ell = 25$. We will use the following test problems

-  *Strong scaling* -- Fixed size problem with varying number of processes. We expect this to start slow and get faster as we increase the number of processes up to a certain point where the overhead of communication will start to deteriorate the computation time.

- *Weak scaling* -- Varying number of processes and varying size of the problem (constant amount of work per node). We expect this to start fast and get slower as we increase the size of the problem as the cost of communication will deteriorate the computation time.

- *SuiteSparse Experiment* -- Varying number of processes on a matrix from the SuiteSparse matrix collection. We chose the matrix `atmosmodd` which has a symmetric pattern of non-zeros but is itself non-symmetric to see how the Hessenberg matrix is affected by the non-symmetry (we get a proper, non tridiagonal hessenberg matrix as with the symmetric Laplacian case).

== Experiment: 3D Laplacian

All the sparse laplacian matrices are generated using the following Julia code, which generates the 3D Laplacian discretization for a grid of side $N$. The matrix is generated sparse using the `spdiagm` function from `SparseArrays`, `kron` is the Kronecker product function from `LinearAlgebra` and `matwrite` is a function from the `MAT` package to write the matrix to a file in the HDF5 format as needed by PETSc.

#align(center, box(fill: luma(93%), radius: 5pt, inset: 5pt)[
  #set text(size: 10pt)
  #set par(leading: 4pt)
  ```jl
  using LinearAlgebra
  using SparseArrays
  using MAT

  N = parse(Int, ARGS[1])

  ex = fill(1, N - 1)
  ey = fill(1, N - 1)
  ez = fill(1, N - 1)

  Dxx = spdiagm(-1 => ex, 0 => -2 * [ex; 1], +1 => ex)
  Dyy = spdiagm(-1 => ey, 0 => -2 * [ey; 1], +1 => ey)
  Dzz = spdiagm(-1 => ez, 0 => -2 * [ez; 1], +1 => ez)

  Ix = spdiagm(0 => [ex; 1])
  Iy = spdiagm(0 => [ey; 1])
  Iz = spdiagm(0 => [ez; 1])

  L = kron(Dxx, Iy, Iz) + kron(Ix, Dyy, Iz) + kron(Ix, Iy, Dzz)

  matwrite("laplacian_$N.mat", Dict("A" => L))
  ```
])

We dropped the factor $1 slash h_i^2$ from the matrix and computed its eigenvalues for a grid of size $N times N times N$. From the previous discussion we can see that the eigenvalues of the 3D Laplacian are given by the following expression for $i_1, i_2, i_3 = 1, dots, N$:
$
  lambda_(i_1, i_2, i_3) =
  - 4 (
    cos^2((pi i_1) / (2(n_1 + 1))) +
    cos^2((pi i_2) / (2(n_2 + 1))) +
    cos^2((pi i_3) / (2(n_3 + 1)))
  )
$

The factor $cos^2(...)$ is bounded in $[0, 1]$ so the eigenvalues are bounded in $[-12, 0]$. First we can check the correctness of the implementation by checking the eigenvalues and eigenvectors returned by our implementation. In @laplacian-hessenberg we can see the resulting Hessenberg matrix while in @laplacian-schur we have the Schur form returned by `dhesqr`.

#figure(
  image("plots/cropped/laplacian-hessenberg.png", width: 70%),
  caption: [
    Hessenberg matrix for the 3D Laplacian with $N = 20$ and $ell = 25$.
  ]
) <laplacian-hessenberg>

#figure(
  image("plots/cropped/laplacian-schur.png", width: 70%),
  caption: [
    Schur matrix for the 3D Laplacian with $N = 20$ and $ell = 25$ after calling `dhesqr`
  ]
) <laplacian-schur>

We get the following eigenvalues:

$
Lambda = {
  && -11.73,
  && -11.43,
  && -11.07,
  && -10.64,
  && -10.13, \
  && -9.55,
  && -8.91,
  && -8.21,
  && -7.47,
  && -6.82, \
  && -6.16,
  && -5.49,
  && -4.81,
  && -4.11,
  && -3.59, \
  && -3.09,
  && -2.64,
  && -2.16,
  && -1.61,
  && -1.12, \
  && -0.91,
  && -0.6,
  && -0.43,
  && -0.24,
  && -0.07#hide([,])
  &&
} 
$

#pagebreak()

=== Strong Scaling

First we tried to run the program starting from a problem size of $N = 20$ and increasing the problem size by doubling each time. As we can see in @laplacian-strong-scaling-3 the performance is initially poor (for $N in {20, 40}$) but starts to improve as we increase the size of the problem as initially the problem is too small and the communication overhead is too high.

#figure(
  image("plots/cropped/laplacian-strong-scaling-3.png", width: 75%),
  caption: [
    Times for varying node count with $N in {20, 40, 80}$, $ell = 25$.
  ]
) <laplacian-strong-scaling-3>

We get our final result in @laplacian-strong-scaling-4 for problem size up to $N = 160$ and we can see that the performance is quite good. The log-log plot in @laplacian-strong-scaling-4-loglog better shows the performance in all the range of problem sizes.

#figure(
  image("plots/cropped/laplacian-strong-scaling-4.png", width: 75%),
  caption: [
    Times for varying node count with $N in {20, 40, 80, 160}$ and $ell = 25$.
  ]
) <laplacian-strong-scaling-4>

#figure(
  image("plots/cropped/laplacian-strong-scaling-4-loglog.png", width: 75%),
  caption: [
    Times log-log for varying node count with $N in {20, 40, 80, 160}$ and $ell = 25$.
  ]
) <laplacian-strong-scaling-4-loglog>

From these results we can also see how the performance stabilizes as we increase the node count and that at a certain point adding more nodes does not improve as much the performance.

// #pagebreak()

=== Weak Scaling

For the weak scaling we generated the matrices for $N = ceil(root(3, i times 20000))$ for $i in {1, ..., 25}$ to get a _constant amount of work per node_. We can see in @laplacian-weak-scaling that the performance degrades as we increase the number of nodes. This is expected as the problem size increases and the communication overhead becomes more significant.

#figure(
  image("plots/cropped/laplacian-weak-scaling.png", width: 75%),
  caption: [
    Times with $N = ceil(root(3, i times 20000)) = {28, ..., 80}$ for $i in {1, ..., 25}$ and $ell = 25$.
  ]
) <laplacian-weak-scaling>

== Experiment: `atmosmodd` from SuiteSparse

We also evaluated the performance against a matrix from the SuiteSparse collection. We choose the `atmosmodd` matrix. It is a non-symmetric matrix with a symmetric pattern of non-zeros, here is some information about the matrix:

#[
  #set align(center)
  #set text(size: 9pt)
  #set table(stroke: 0.5pt)

  #v(1.5em, weak: true)
  #pad(x: -50%, grid(
    align: center, 
    columns: 2,
    gutter: 1.5em,
    table(
      columns: (auto, auto),
      align: left,
      inset: 3pt,
      [*Name*], [atmosmodd],
      [*Group*], [Bourchtein],
      [*Matrix ID*], [2265],
      [*Num Rows*], [$1 quote.single 270 quote.single 432$],
      [*Num Cols*], [$1 quote.single 270 quote.single 432$],
      [*Non-zeros*], [$8 quote.single 814 quote.single 880$],
      [*Pattern Entries*], [$8 quote.single 814 quote.single 880$],
      [*Kind*], [Computational Fluid Dynamics Problem],
      [*Symmetric*], [No],
      [*Date*], [$2009$],
      [*Author*], [A. Bourchtein],
      [*Editor*], [T. Davis]
    ),
    table(
      columns: (auto, auto),
      align: left,
      inset: 3pt,
      [*Structural Rank*], [$1 quote.single 270 quote.single 432$],
      [*Structural Rank Full*], [true],
      [*Num Dmperm Blocks*], [$1$],
      [*Strongly Connect Components*], [$1$],
      [*Num Explicit Zeros*], [$0$],
      [*Pattern Symmetry*], [$100%$],
      [*Numeric Symmetry*], [$66.9%$],
      [*Cholesky Candidate*], [No],
      [*Positive Definite*], [No],
      [*Type*], [Real]
    )
  ))
  #v(1.5em, weak: true)
]

In @atmosmodd-hessenberg we can see that the resulting Hessenberg matrix looks tridiagonal but when looking at the log-log plot we actually see that the matrix is a proper Hessenberg matrix.

The resulting Schur matrix is shown in @atmosmodd-schur and the computed eigenvalues are the following:

$
  Lambda = {
    && -187 quote.single 078.67,
    && -181 quote.single 294.17,
    && -173 quote.single 634.23,
    && -167 quote.single 725.76,
    && -160 quote.single 474.49, \
    && -152 quote.single 024.13,
    && -142 quote.single 505.34,
    && -132 quote.single 037.86,
    && -120 quote.single 709.76,
    && -108 quote.single 473.08, \
    && -93 quote.single 768.3,
    && -83 quote.single 850.61,
    && -75 quote.single 095.74,
    && -65 quote.single 634.56,
    && -56 quote.single 074.01, \
    && -46 quote.single 728.52,
    && -37 quote.single 822.94,
    && -29 quote.single 538.87,
    && -22 quote.single 025.98,
    && -15 quote.single 371.79, \
    && -9 quote.single 907.26,
    && -5 quote.single 664.09,
    && -2 quote.single 596.04,
    && -739.49,
    && -60.07#hide([,])
    &&
  }
$

#figure(
  image("plots/cropped/suitesparse-atmosmodd-hessenberg.png", width: 70%),
  caption: [
    Hessenberg matrix for the SuiteSparse matrix `atmosmodd` with $ell = 25$.
  ]
) <atmosmodd-hessenberg>

#figure(
  image("plots/cropped/suitesparse-atmosmodd-schur.png", width: 70%),
  caption: [
    Schur matrix for the SuiteSparse matrix `atmosmodd` with $ell = 25$ after calling `dhesqr`
  ]
) <atmosmodd-schur>

Finally, we can see the same strong scaling performance for this non-symmetric matrix as for symmetric the 3D Laplacian.

#figure(
  image("plots/cropped/suitesparse-atmosmodd.png", width: 75%),
  caption: [
    Times of the Arnoldi method for the SuiteSparse matrix `atmosmodd` with $ell = 25$ with varying number of nodes.
  ]
) <atmosmodd-varying-nodecount>