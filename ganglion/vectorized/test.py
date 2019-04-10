import numpy as np 

def fillitold(a, b, w, cstride, rstride):
    a_rows = a.shape[0]
    a_cols = a.shape[1]
    b_rows = b.shape[0]
    b_cols = b.shape[1]

    num_strides_col = (a_cols - b_cols) / cstride + 1
    num_strides_row = (a_rows - b_rows) / rstride + 1

    expected_shape = (num_strides_row, num_strides_col)

    if a_rows < b_rows:
        raise ValueError("The number of rows in matrix B must be greater than or equal to the number of rows in matrix A")
    if a_cols < b_cols:
        raise ValueError("The number of columns in matrix B must be greater than or equal to the number of columns in matrix A")
    if not num_strides_col.is_integer():
        raise ValueError("Uneven column stride")
    if not num_strides_row.is_integer():
        raise ValueError("Uneven row stride")
    if b.shape != expected_shape:
        raise ValueError("Expected shape of matrix B does not match expected shape of matrix A")
    num_strides_row = int(num_strides_row)
    num_strides_col = int(num_strides_col)

    w_a_keys = np.reshape(np.arange(0, a_rows * a_cols, 1), (a_rows, a_cols))
    w_b_keys = np.reshape(np.arange(0, b_rows * b_cols, 1), (b_rows, b_cols))

    for r_s in range(num_strides_row):
        r_index = r_s * rstride  # row index in A
        for c_s in range(num_strides_col):
            c_index = c_s * cstride  # col index in A

            a_mod = w_a_keys[r_index: r_index + b_rows, c_index: c_index + b_cols].flatten()
            
            w[a_mod, w_b_keys[r_s, c_s]] = 1.0

    
    return w


def fillit(a, b, patch, w, cstride, rstride):
    a_rows = a.shape[0]
    a_cols = a.shape[1]
    b_rows = b.shape[0]
    b_cols = b.shape[1]
    patch_rows = patch.shape[0]
    patch_cols = patch.shape[1]

    num_strides_col = (a_cols - patch_cols) / cstride + 1
    num_strides_row = (a_rows - patch_rows) / rstride + 1

    
    if a_rows < b_rows:
        raise ValueError("The number of rows in matrix B must be greater than or equal to the number of rows in matrix A")
    if a_cols < b_cols:
        raise ValueError("The number of columns in matrix B must be greater than or equal to the number of columns in matrix A")
    if not num_strides_col.is_integer():
        raise ValueError("Uneven column stride")
    if not num_strides_row.is_integer():
        raise ValueError("Uneven row stride")
    num_strides_row = int(num_strides_row)
    num_strides_col = int(num_strides_col)
    expected_shape = (num_strides_row, num_strides_col)
    if b.shape != expected_shape:
        raise ValueError("Expected shape of matrix B does not match expected shape of matrix A")
    

    w_a_keys = np.reshape(np.arange(0, a_rows * a_cols, 1), (a_rows, a_cols))
    w_b_keys = np.reshape(np.arange(0, b_rows * b_cols, 1), (b_rows, b_cols))

    for r_s in range(num_strides_row):
        r_index = r_s * rstride  # row index in A
        for c_s in range(num_strides_col):
            c_index = c_s * cstride  # col index in A

            a_mod = w_a_keys[r_index: r_index + patch_rows, c_index: c_index + patch_cols].flatten()
            
            w[a_mod, w_b_keys[r_s, c_s]] = 1.0

    
    return w


    

    
layer1 = np.zeros((5,5), dtype=np.float)
layer2 = np.zeros((2,2), dtype=np.float)
w = np.zeros((layer1.shape[0]*layer1.shape[1], layer2.shape[0]*layer2.shape[1]), dtype=np.float)

print(layer1.shape)
print(layer2.shape)
print(w.shape)

w = fillit(layer1, layer2, np.zeros((3,3)), w, 2, 2)
print(w)