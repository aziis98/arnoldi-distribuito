#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

static char help[] = "Example PETSc program\n\n";

// extern PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void *);
// extern PetscErrorCode ComputeRHS(KSP, Vec, void *);
// extern PetscErrorCode ComputeInitialSolution(DM, Vec);

PetscErrorCode ArnoldiIteration(Mat A, Vec b, PetscInt n, Vec *Q, Mat H);

int main(int argc, char **argv) {
    Vec b;
    PetscInt n, l;

    PetscFunctionBeginUser;
    PetscInitialize(&argc, &argv, (char *)0, help);

    PetscBool flg;
    PetscOptionsGetInt(NULL, NULL, "-n", &n, &flg);
    if (!flg)
        n = 176;

    PetscOptionsGetInt(NULL, NULL, "-l", &l, &flg);
    if (!flg)
        l = 4;

    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, n);
    VecSetType(b, VECMPI);

    VecSet(b, 1.0);
    // VecSetValue(b, 0, 1.0, INSERT_VALUES);

    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    // MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    // MatSetType(A, MATMPIAIJ);
    PetscViewer v;
    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &v));
    PetscCall(PetscViewerSetType(v, PETSCVIEWERHDF5));
    PetscCall(PetscViewerPushFormat(v, PETSC_VIEWER_HDF5_MAT));
    PetscCall(PetscViewerSetFromOptions(v));
    PetscCall(PetscViewerFileSetMode(v, FILE_MODE_READ));
    PetscCall(PetscViewerFileSetName(
        v, "../matrices/laplacian/laplacian-discretization-3d.mat"));

    PetscCall(MatSetOptionsPrefix(A, "a_"));
    PetscCall(PetscObjectSetName((PetscObject)A, "A"));

    // PetscCall(
    //     PetscOptionsGetString(NULL, NULL, "-f", name, sizeof(name), &flg));
    // PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_SUP,
    //            "Must provide a binary file for the matrix");

    // // PetscCall(MatLoad(A, v));
    // PetscCall(PetscViewerBinaryOpen(
    //     PETSC_COMM_WORLD,
    //     "../matrices/laplacian/laplacian-discretization-3d.mat",
    //     FILE_MODE_READ, &v));
    PetscCall(MatLoad(A, v));

    // // A := diag(-1, 2, -1)
    // for (PetscInt i = 0; i < n; i++) {
    //     PetscScalar v[3] = {-1.0, 2.0, -1.0};
    //     PetscInt col[3] = {i - 1, i, i + 1};
    //     PetscInt ncol = 0;
    //     if (i > 0) {
    //         col[ncol] = i - 1;
    //         v[ncol] = -1.0;
    //         ncol++;
    //     }
    //     col[ncol] = i;
    //     v[ncol] = 2.0;
    //     ncol++;
    //     if (i < n - 1) {
    //         col[ncol] = i + 1;
    //         v[ncol] = -1.0;
    //         ncol++;
    //     }
    //     MatSetValues(A, 1, &i, ncol, col, v, INSERT_VALUES);
    //     // MatSetValue(A, i, i, (PetscScalar)(i + 1), INSERT_VALUES);
    // }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // MatView(A, PETSC_VIEWER_DRAW_WORLD);
    // MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    // VecView(b, PETSC_VIEWER_DRAW_WORLD);
    // VecView(b, PETSC_VIEWER_STDOUT_WORLD);

    printf("[Arnoldi] Allocating memory for Krylov subspace basis\n");

    Vec *Q;
    PetscMalloc1(l, &Q);

    for (PetscInt i = 0; i < l; i++) {
        VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &Q[i]);
    }

    printf("[Arnoldi] Constructing Hessenberg matrix\n");

    Mat H;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &H));
    PetscCall(MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, l + 1, l));
    PetscCall(MatSetType(H, MATDENSE));
    // MatSetType(H, MATMPIAIJ);

    printf("[Arnoldi] Starting iteration\n");

    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));

    PetscCall(ArnoldiIteration(A, b, l, Q, H));

    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

    printf("[Arnoldi] Done\n");

    // print Hessenberg matrix to file

    PetscViewer v2;
    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &v2));
    PetscCall(PetscViewerSetType(v2, PETSCVIEWERHDF5));
    PetscCall(PetscViewerPushFormat(v2, PETSC_VIEWER_HDF5_MAT));
    PetscCall(PetscViewerFileSetMode(v2, FILE_MODE_WRITE));
    PetscCall(PetscViewerFileSetName(v2, "hessenberg.mat"));
    PetscCall(MatView(H, v2));

    // for (PetscInt i = 0; i < l + 1; i++) {
    //     VecView(Q[i], PETSC_VIEWER_STDOUT_WORLD);
    // }

    // MatView(H, PETSC_VIEWER_STDOUT_WORLD);

    // for (PetscInt i = 0; i < l + 1; i++) {
    //     PetscCall(VecDestroy(&Q[i]));
    // }

    // PetscCall(PetscFree(Q));

    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    PetscFinalize();
    return 0;
}

PetscErrorCode ArnoldiIteration(Mat A, Vec b, PetscInt n, Vec *Q, Mat H) {
    PetscFunctionBeginUser;

    PetscScalar eps = 1e-12;
    PetscInt m;
    PetscCall(VecGetSize(b, &m));

    Vec q;

    PetscCall(MatZeroEntries(H));

    PetscCall(VecDuplicate(b, &q));
    PetscCall(VecCopy(b, q));
    PetscCall(VecNormalize(q, NULL));

    Q[0] = q;

    for (PetscInt k = 1; k < n + 1; k++) {
        // printf("[Arnoldi] Iteration %d\n", k);

        Vec v;
        PetscCall(VecDuplicate(b, &v));
        PetscCall(MatMult(A, Q[k - 1], v));

        // Reorthogonalization using modified Gram-Schmidt
        for (PetscInt j = 0; j < k; j++) {
            // printf("[Arnoldi] Reorthogonalization %d\n", j);

            PetscScalar h;
            PetscCall(VecDot(Q[j], v, &h));
            PetscCall(MatSetValue(H, j, k - 1, h, INSERT_VALUES));
            PetscCall(VecAXPY(v, -h, Q[j]));
        }

        // Normalize
        PetscScalar h;
        PetscCall(VecNorm(v, NORM_2, &h));
        PetscCall(MatSetValue(H, k, k - 1, h, INSERT_VALUES));

        // Check for convergence
        if (h > eps) {
            PetscCall(VecNormalize(v, NULL));
            Q[k] = v;
        } else {
            break;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
