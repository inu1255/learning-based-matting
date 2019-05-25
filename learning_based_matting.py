import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys

from numpy.lib.stride_tricks import as_strided


def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def learning_based_matte(img, trimap, c=800, mylambda=0.0000001):
    # 前景标记 table
    foreground = trimap == 255
    # 背景标记 table
    background = trimap == 0
    # 前景:1 背景:-1 没有标记:0
    mask = np.zeros(trimap.shape)
    mask[foreground] = 1
    mask[background] = -1

    L = getLapFast(img, mask, mylambda)
    C = getC(mask, c)
    alpha = solveQurdOpt(L, C, mask)
    return alpha


def getLapFast(img, mask, mylambda=0.0000001, win_rad=1):
    # 窗口长度: 3
    w_s = win_rad*2 +1
    # 窗口面积: 9
    win_size = (w_s)**2
    # 转换为 0~1
    img = img/255
    h, w, c = img.shape
    # ds表 h*w
    indsM = np.reshape(np.arange(h*w), (h, w))
    # 扁平图 
    ravelImg = img.reshape(h*w, c)

    # 草图标记  草图位置为True
    scribble_mask = mask != 0
    # 需要训练的像素点个数 窗口覆盖区域，没有标记的像素数量
    numPix4Training = np.sum(1-scribble_mask[win_rad:-win_rad, win_rad:-win_rad])
    # 
    numNonzeroValue = numPix4Training*win_size**2

    row_inds = np.zeros(numNonzeroValue)
    col_inds = np.zeros(numNonzeroValue)
    vals = np.zeros(numNonzeroValue)

    # 遍历表，及对应的
    win_indsMat = rolling_block(indsM, block=(w_s, w_s))
    win_indsMat = win_indsMat.reshape(h - 2*win_rad, w - 2*win_rad, win_size)

    t = 0
    # Repeat on each legal pixel
    # Cannot fully vectorize since the size of win_inds varies by the number of unknown pixels
    # If we preform the computation for all pixels in fully vectorised form it's slower.
    for i in range(h - 2*win_rad):
        win_inds = win_indsMat[i, :]
        # 当前行，没有标记的列 3*3窗口对应的坐标
        win_inds = win_inds[np.logical_not(scribble_mask[i+win_rad, win_rad:w-win_rad])]
        # 当前行，窗口对应的像素 m*9*4
        winI = ravelImg[win_inds]
        # 宽度
        m = winI.shape[0]
        # 当前行，窗口对应像素~1
        winI = np.concatenate((winI, np.ones((m, win_size, 1))), axis =2)
        I = np.tile(np.eye(win_size), (m, 1, 1))
        # m*9*9
        I[:, -1, -1] = 0
        # 窗口矩阵相乘->m*9*9
        winITProd = np.einsum('...ij,...kj ->...ik', winI, winI)
        # 强化
        fenmu = winITProd + mylambda*I
        # 求逆
        invFenmu = np.linalg.inv(fenmu)
        F = np.einsum('...ij,...jk->...ik', winITProd, invFenmu)
        I_F = np.eye(win_size) - F
        lapcoeff = np.einsum('...ji,...jk->...ik', I_F, I_F)

        vals[t: t+(win_size**2)*m] = lapcoeff.ravel()
        row_inds[t:t+(win_size**2)*m] = np.repeat(win_inds, win_size).ravel()
        col_inds[t:t+(win_size**2)*m] = np.tile(win_inds, win_size).ravel()
        t = t+(win_size**2)*m
    L = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(h*w, h*w))
    return L


def solveQurdOpt(L, C, alpha_star):
    mylambda = 1e-6
    D = scipy.sparse.eye(L.shape[0])

    alpha = scipy.sparse.linalg.spsolve(L + C + D*mylambda, C @ alpha_star.ravel())
    alpha = np.reshape(alpha, alpha_star.shape)

    # if alpha value of labelled pixels are -1 and 1, the resulting alpha are
    # within [-1 1], then the alpha results need to be mapped to [0 1]
    if np.min(alpha_star.ravel()) == -1:
        alpha = alpha*0.5+0.5
    alpha = np.maximum(np.minimum(alpha, 1), 0)
    return alpha


def getC(mask, c=800):
    scribble_mask = (mask != 0).astype(int)
    C = scipy.sparse.diags(c * scribble_mask.ravel())
    return C


def main():
    img = scipy.misc.imread(sys.argv[1])
    trimap = scipy.misc.imread(sys.argv[2], flatten='True')

    alpha = learning_based_matte(img, trimap)
    scipy.misc.imsave(sys.argv[3], alpha)
    # plt.imshow(alpha, cmap = 'gray')
    # plt.show()

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import scipy.misc
    main()
