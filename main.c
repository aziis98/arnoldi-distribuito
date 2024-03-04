static char help[] = "Solves a tridiagonal linear system.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b, u;                                   /* approx solution, RHS, exact solution */
  Mat         A;                                         /* linear system matrix */
  KSP         ksp;                                       /* linear solver context */
  PC          pc;                                        /* preconditioner context */
  PetscReal   norm, tol = 1000. * PETSC_MACHINE_EPSILON; /* norm of solution error */
  PetscInt    i, n      = 10, col[3], its, rstart, rend, nlocal;
  PetscScalar one = 1.0, value[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));

  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(VecGetLocalSize(x, &nlocal));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, nlocal, nlocal, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  if (!rstart) {
    rstart   = 1;
    i        = 0;
    col[0]   = 0;
    col[1]   = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  }
  if (rend == n) {
    rend     = n - 1;
    i        = n - 1;
    col[0]   = n - 2;
    col[1]   = n - 1;
    value[0] = -1.0;
    value[1] = 2.0;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  }

  value[0] = -1.0;
  value[1] = 2.0;
  value[2] = -1.0;
  for (i = rstart; i < rend; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(VecSet(u, one));
  PetscCall(MatMult(A, u, b));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  PetscCall(KSPSetOperators(ksp, A, A));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1.e-7, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}
