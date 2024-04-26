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
  Mat A;
  Vec b;
  PetscInt n, l;

  PetscFunctionBeginUser;
  PetscInitialize(&argc, &argv, (char *)0, help);

  PetscBool flg;
  PetscOptionsGetInt(NULL, NULL, "-n", &n, &flg);
  if (!flg)
    n = 10;

  PetscOptionsGetInt(NULL, NULL, "-l", &l, &flg);
  if (!flg)
    l = 4;

  VecCreate(PETSC_COMM_WORLD, &b);
  VecSetSizes(b, PETSC_DECIDE, n);
  VecSetType(b, VECMPI);

  VecSet(b, 1.0);
  // VecSetValue(b, 0, 1.0, INSERT_VALUES);

  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);

  MatSetType(A, MATMPIAIJ);

  // A := diag(-1, 2, -1)
  for (PetscInt i = 0; i < n; i++) {
    // PetscScalar v[3] = {-1.0, 2.0, -1.0};
    // PetscInt col[3] = {i - 1, i, i + 1};
    // PetscInt ncol = 0;
    // if (i > 0) {
    //   col[ncol] = i - 1;
    //   v[ncol] = -1.0;
    //   ncol++;
    // }
    // col[ncol] = i;
    // v[ncol] = 2.0;
    // ncol++;
    // if (i < n - 1) {
    //   col[ncol] = i + 1;
    //   v[ncol] = -1.0;
    //   ncol++;
    // }
    // MatSetValues(A, 1, &i, ncol, col, v, INSERT_VALUES);
    MatSetValue(A, i, i, (PetscScalar)(i + 1), INSERT_VALUES);
  }

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  // MatView(A, PETSC_VIEWER_DRAW_WORLD);
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(b, PETSC_VIEWER_DRAW_WORLD);
  VecView(b, PETSC_VIEWER_STDOUT_WORLD);

  printf("Allocating memory for Krylov subspace basis\n");

  Vec *Q;
  PetscMalloc1(l, &Q);

  for (PetscInt i = 0; i < l; i++) {
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &Q[i]);
  }

  printf("Constructing Hessenberg matrix\n");

  Mat H;
  MatCreate(PETSC_COMM_WORLD, &H);
  MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, l + 1, l);
  // MatSetType(H, MATMPIAIJ);
  MatSetType(H, MATDENSE);

  printf("Starting Arnoldi iteration\n");

  MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  PetscCall(ArnoldiIteration(A, b, l, Q, H));
  MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

  for (PetscInt i = 0; i < l + 1; i++) {
    VecView(Q[i], PETSC_VIEWER_STDOUT_WORLD);
  }

  MatView(H, PETSC_VIEWER_STDOUT_WORLD);

  for (PetscInt i = 0; i < l + 1; i++) {
    VecDestroy(&Q[i]);
  }

  // PetscFree(Q);

  MatDestroy(&A);
  VecDestroy(&b);
  PetscFinalize();
  return 0;
}

PetscErrorCode ArnoldiIteration(Mat A, Vec b, PetscInt n, Vec *Q, Mat H) {
  PetscFunctionBeginUser;

  PetscScalar eps = 1e-12;
  PetscInt m;
  VecGetSize(b, &m);

  Vec q;

  MatZeroEntries(H);

  VecDuplicate(b, &q);
  VecCopy(b, q);
  VecNormalize(q, NULL);

  Q[0] = q;

  for (PetscInt k = 1; k < n + 1; k++) {
    Vec v;
    VecDuplicate(b, &v);
    MatMult(A, Q[k - 1], v);

    for (PetscInt j = 0; j < k; j++) {
      PetscScalar h;
      VecDot(Q[j], v, &h);
      MatSetValue(H, j, k - 1, h, INSERT_VALUES);
      VecAXPY(v, -h, Q[j]);
    }

    PetscScalar h;
    VecNorm(v, NORM_2, &h);
    MatSetValue(H, k, k - 1, h, INSERT_VALUES);

    if (h > eps) {
      VecNormalize(v, NULL);
      Q[k] = v;
    } else {
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
