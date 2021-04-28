import numpy as np
cimport numpy as np
cimport cython

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def im2col(DTYPE_t[:,:,:,:] X, int N, int channel, int height, int width, 
           int Fheight, int Fwidth, int pad_h=0, int pad_w=0, 
           int stride_height=1, int stride_width=1):

    cdef int out_height = (height + 2 * pad_h - Fheight) / stride_height + 1
    cdef int out_width  = (width + 2 * pad_w - Fwidth) / stride_width + 1

    cdef DTYPE_t[:,:,:,:] X_padded = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h),
                            (pad_w, pad_w)), mode='constant')

    cdef DTYPE_t[:,:] X_col = np.zeros((channel * Fheight * Fwidth, N * out_height * out_width))

    im2col_inner(X_col, X_padded, N, channel, height, width, out_height, out_width, Fheight, Fwidth, 
                 stride_height, stride_width)

    return X_col


def col2im(DTYPE_t[:,:] X_col, int N, int channel, int height, int width,
           int Fheight, int Fwidth, int pad_h=0, int pad_w=0, int stride_height=1,
           int stride_width=1):

    cdef int out_height = (height + 2 * pad_h - Fheight) / stride_height + 1
    cdef int out_width = (width + 2 * pad_w - Fwidth) / stride_width + 1
    cdef int padded_height = height + 2 * pad_h
    cdef int padded_width = width + 2 * pad_w

    cdef DTYPE_t[:,:,:,:] X_padded = np.zeros((N, channel, padded_height, padded_width))

    col2im_inner(X_col, X_padded, N, channel, height, width, out_height, out_width, Fheight, Fwidth,
                 stride_height, stride_width)

    if pad_h > 0 and pad_w > 0:
        return X_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        return X_padded[:, :, pad_h:-pad_h, :]
    elif pad_w > 0:
        return X_padded[:, :, :, pad_w:-pad_w]
    
    return X_padded


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_inner(DTYPE_t[:,:] X_col,
                      DTYPE_t[:,:,:,:] X_padded,
                      int N, int channel, int height, int width,
                      int out_height, int out_width,
                      int Fheight, int Fwidth, int stride_height, int stride_width) except? -1:

    cdef int i, c, cur_block_h, cur_block_w, block_row, block_col, row, col

    
    for c in range(channel):
        for cur_block_h in range(out_height):
            for cur_block_w in range(out_width):
                for block_row in range(Fheight):
                    for block_col in range(Fwidth):
                        row = c * Fheight * Fwidth + block_row * Fwidth + block_col

                        for i in range(N):
                            col = i * out_height * out_width + cur_block_h * out_width + cur_block_w
                            
                            X_col[row, col] = X_padded[i, c, cur_block_h * stride_height + block_row, 
                                                       cur_block_w * stride_width + block_col]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_inner(DTYPE_t[:,:] X_col,
                      DTYPE_t[:,:,:,:] X_padded,
                      int N, int channel, int height, int width,
                      int out_height, int out_width,
                      int Fheight, int Fwidth, int stride_height, int stride_width) except? -1:

    cdef int i, c, cur_block_h, cur_block_w, block_row, block_col, row, col

    
    for c in range(channel):
        for cur_block_h in range(out_height):
            for cur_block_w in range(out_width):
                for block_row in range(Fheight):
                    for block_col in range(Fwidth):
                        row = c * Fheight * Fwidth + block_row * Fwidth + block_col
                        
                        for i in range(N):
                            col = i * out_height * out_width + cur_block_h * out_width + cur_block_w

                            X_padded[i, c, cur_block_h * stride_height + block_row, 
                                    cur_block_w * stride_width + block_col] += X_col[row, col]