# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp
from scipy.sparse import issparse



def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None

    if A.ndim != 2:
        return None

    rows, cols = A.shape
    if rows != cols:
        return None
    
    if not np.issubdtype(A.dtype, np.number):
        return None

    if issparse(A):
        A = A.toarray()

    
    diag_abs = np.abs(np.diag(A))


    row_abs_sums = np.sum(np.abs(A), axis=1) - diag_abs

    return bool(np.all(diag_abs > row_abs_sums))



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:

    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    
    if x.ndim != 1 or b.ndim != 1:
        return None

    m, n = A.shape
    if n != x.shape[0] or m != b.shape[0]:
        return None

    residual = b - A @ x
    res_norm = np.linalg.norm(residual)

    return float(res_norm)
