#include <lapack.h>

#include <lapacke.h>
#include <mpi.h>
#include <petscconf.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscviewertypes.h>
#include <stdlib.h>

static char help[] = "Example PETSc program\n\n";

void swap(double *a, double *b) {
    double t = *a;
    *a = *b;
    *b = t;
}

// int compare_double(const void *a, const void *b) {
//     int aa = *(int *)a;
//     int bb = *(int *)b;

//     if (aa < bb)
//         return -1;
//     if (aa > bb)
//         return 1;
//     return 0;
// }

PetscErrorCode ArnoldiIteration(Mat A, Vec b, PetscInt n, PetscInt m, Vec *Q, double *h);

int main(int argc, char **argv) {
    PetscInt n, l;
    char matrix_name[PETSC_MAX_PATH_LEN];

    PetscFunctionBeginUser;
    PetscInitialize(&argc, &argv, (char *)0, help);

    PetscBool flg;
    PetscOptionsGetInt(NULL, NULL, "-n", &n, &flg);
    if (!flg)
        n = -1;

    PetscOptionsGetInt(NULL, NULL, "-l", &l, &flg);
    if (!flg)
        l = 10;

    PetscOptionsGetString(NULL, NULL, "-matrix_path", matrix_name, sizeof(matrix_name), &flg);
    if (!flg)
        snprintf(matrix_name, sizeof(matrix_name), "./matrices/laplacian/laplacian-discretization-3d.mat");

    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    PetscViewer v;
    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &v));
    PetscCall(PetscViewerSetType(v, PETSCVIEWERHDF5));
    PetscCall(PetscViewerPushFormat(v, PETSC_VIEWER_HDF5_MAT));
    PetscCall(PetscViewerSetFromOptions(v));
    PetscCall(PetscViewerFileSetMode(v, FILE_MODE_READ));
    PetscCall(PetscViewerFileSetName(v, "./matrices/laplacian/laplacian-discretization-3d.mat"));

    PetscCall(MatSetOptionsPrefix(A, "a_"));
    PetscCall(PetscObjectSetName((PetscObject)A, "A"));

    PetscCall(MatLoad(A, v));

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // TIME: Assembly construction

    if (n == -1) {
        MatGetSize(A, &n, NULL);
    }

    Vec b;
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, n);
    VecSetType(b, VECMPI);

    VecSet(b, 1.0);
    // VecSetValue(b, 0, 1.0, INSERT_VALUES);

    // MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    // VecView(b, PETSC_VIEWER_STDOUT_WORLD);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Arnoldi] Allocating memory for Krylov subspace basis\n"));

    Vec *Q;
    PetscMalloc1(l, &Q);

    for (PetscInt i = 0; i < l; i++) {
        VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &Q[i]);
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Arnoldi] Constructing Hessenberg matrix\n"));

    // Mat H;
    // PetscCall(MatCreate(PETSC_COMM_SELF, &H));
    // PetscCall(PetscObjectSetName((PetscObject)H, "hessenberg"));
    // PetscCall(MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, l + 1, l));
    // PetscCall(MatSetType(H, MATDENSE));

    // PetscCall(MatCreate(PETSC_COMM_WORLD, &H));

    // MatSetType(H, MATMPIAIJ);

    double *H = (double *)malloc((l + 1) * l * sizeof(double));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Arnoldi] Starting iteration\n"));

    // TIME:START

    PetscCall(ArnoldiIteration(A, b, l, n, Q, H));

    // PetscCall(MatSetValue(H, 2, 3, -1, INSERT_VALUES));
    // PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    // PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Arnoldi] Done\n"));

    int rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    if (rank == 0) {
        // print Hessenberg matrix
        printf("H = \n");
        for (int i = 0; i < l + 1; i++) {
            for (int j = 0; j < l; j++) {
                printf("%.3f ", H[i * l + j]);
            }
            printf("\n");
        }

        // call LAPACK function "DHSEQR" to compute the eigenvalues of the Hessenberg matrix

        double *wr = (double *)malloc(l * sizeof(double));
        double *wi = (double *)malloc(l * sizeof(double));
        double *z = (double *)malloc(l * l * sizeof(double));
        double *work = (double *)malloc(3 * l * sizeof(double));
        int info;

        LAPACKE_dhseqr(LAPACK_ROW_MAJOR, 'E', 'I', l, 1, l, H, l, wr, wi, z, l);

        // TIME:END

        // sort eigenvalues
        for (int i = 0; i < l; i++) {
            for (int j = i + 1; j < l; j++) {
                if (wr[i] > wr[j]) {
                    swap(&wr[i], &wr[j]);
                    swap(&wi[i], &wi[j]);
                    for (int k = 0; k < l; k++) {
                        swap(&z[i * l + k], &z[j * l + k]);
                    }
                }
            }
        }

        // print eigenvalues
        printf("Eigenvalues = \n");
        for (int i = 0; i < l; i++) {
            printf("%.3f + %.3f i\n", wr[i], wi[i]);
        }
    }

    // print eigenvectors
    // printf("Eigenvectors = \n");
    // for (int i = 0; i < l; i++) {
    //     for (int j = 0; j < l; j++) {
    //         printf("%f ", z[i * l + j]);
    //     }
    //     printf("\n");
    // }

    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    // PetscCall(MatDestroy(&H));

    PetscFinalize();
    return 0;
}

PetscErrorCode ArnoldiIteration(Mat A, Vec b, PetscInt n, PetscInt m, Vec *Q, double *h) {
    PetscFunctionBeginUser;

    int rank, total;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &total);

    char *hostname;
    PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
    printf("Process %d of %d\n", rank, total);

    PetscScalar eps = 1e-12;

    Vec q;

    for (PetscInt i = 0; i < n + 1; i++) {
        for (PetscInt j = 0; j < n; j++) {
            h[i * n + j] = 0.0;
        }
    }

    PetscCall(VecDuplicate(b, &q));
    PetscCall(VecCopy(b, q));
    PetscCall(VecNormalize(q, NULL));

    Q[0] = q;

    for (PetscInt k = 1; k < n + 1; k++) {

        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Arnoldi] Iteration %d\n", k));

        Vec v;
        PetscCall(VecDuplicate(b, &v));
        PetscCall(MatMult(A, Q[k - 1], v));

        // Reorthogonalization using modified Gram-Schmidt
        for (PetscInt j = 0; j < k; j++) { // anche solo 3
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Arnoldi] Reorthogonalization %d %d\n", k, j));

            PetscScalar h_ij;
            PetscCall(VecDot(Q[j], v, &h_ij));

            // PetscCall(MatSetValue(H, j, k - 1, h, INSERT_VALUES));
            h[j * n + k - 1] = h_ij;

            PetscCall(VecAXPY(v, -h_ij, Q[j]));
        }

        // Normalize
        PetscScalar h_ij;
        PetscCall(VecNorm(v, NORM_2, &h_ij));

        // PetscCall(MatSetValue(H, k, k - 1, h, INSERT_VALUES));
        h[k * n + k - 1] = h_ij;

        // Check for convergence
        if (h_ij > eps) {
            PetscCall(VecNormalize(v, NULL));
            Q[k] = v;
        } else {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Arnoldi] Early breakdown"));
            break;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
