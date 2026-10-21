// C entry point for ctypes around the official CEC test-function code.
// The official file declares these globals "extern"; its own main.cpp normally defines them.
#include <stdlib.h>

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;

void ENTRY(double *, double *, int, int, int);   // cec17_test_func or cec22_test_func (-DENTRY=...)

extern "C" void cec_eval(double *x, double *f, int nx, int mx, int func_num)
{
    ENTRY(x, f, nx, mx, func_num);
}
