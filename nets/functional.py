import numpy as np


def linear_forward(x, w, b):
    # N, I, J
    out = np.einsum("nj,ij->ni", x, w) + b
    cache = (x, w)
    return out, cache


def linear_backward(dL_dout, cache):
    """
    x: (N, J)
    w: (I, J)
    b: (I,)
    dL_dout: (N, I)

    Compute the derivative ignoring the batch dim:
        out = w @ x
        out_i = w_i1 * x_1 + ... + w_ij * x_j + ... + w_iJ * x_J

    Derivative wrt x:
        dout_i / dx_j = w_ij
        dL / dx_j = sum_i [(dL / dout_i) * (dout_i / dx_j)]
        dL / dx_j = sum_i [(dL / dout_i) * w_ij)]

        In Einstein summation notation, this is
        dL / dx_j = (dL / dout_i) * w_ij
        As the sum is over i implicitly.

        Converting this to einsum means taking the subscripts, and moving them
        Into the first argument:
        np.einsum(i,ij->j, (dL / dout), w)

        Now, we must broadcast across the batch dimension:
        np.einsum(ni,ij->nj, (dL / dout), w)


    Derivative wrt w:
        dout_i / dw_kj = x_j * delta(k,i)
        dL / dw_kj = sum_i [(dL / dout_i) (dout_i / dw_kj)]
        = sum_i [(dL / dout_i) x_j * delta(k,i))]
        = (dL / dout_i) * x_j
        np.einsum("i,j->ij")

        Now, we must sum across the batch dimension:
        np.einsum("ni,nj->ij")

    Derivative wrt b:
        dL / db_i = (dL / dout_i) * (dout_i / db_i)
        = dL / dout_i

        Now, we must sum across the batch dim:
        np.einsum("ni->i", dL_dout)
    """
    x, w = cache
    dL_dx = np.einsum("ni,ij->nj", dL_dout, w)
    dL_dw = np.einsum("ni,nj->ij", dL_dout, x)
    dL_db = np.sum(dL_dout, axis=0)
    return dL_dx, dL_dw, dL_db


def conv2d_forward(x, kernel, b, pad=0, stride=1, num_groups=1):
    """
    FLOPS:
        2 * out_H * out_W * C_in * C_out * K * K
        If there is bias, add N * C_out * out_H * out_W operations
    """
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, kH, kW = kernel.shape

    # num_groups, C_out_per_group, C_in_per_group, kH, kW
    kernel_grouped = kernel.reshape(num_groups, -1, C_in_per_group, kH, kW)
    C_out_per_group = kernel_grouped.shape[1]

    # Pad Input
    Hp = H + 2 * pad
    Wp = W + 2 * pad
    xp = np.zeros((N, C_in, Hp, Wp))
    xp[:, :, pad : pad + H, pad : pad + W] = x

    # Group input by channels
    xp_grouped = xp.reshape(N, num_groups, C_in_per_group, Hp, Wp)

    # Loop over output
    outH = int(np.ceil((Hp - (kH - 1)) / stride))
    outW = int(np.ceil((Wp - (kW - 1)) / stride))
    output_grouped = np.zeros((N, num_groups, C_out_per_group, outH, outW))

    for h in range(output_grouped.shape[-2]):
        for w in range(output_grouped.shape[-1]):
            patch = xp_grouped[
                ..., stride * h : stride * h + kH, stride * w : stride * w + kW
            ]
            output_grouped[:, :, :, h, w] += np.einsum(
                "ngiyx,goiyx->ngo", patch, kernel_grouped
            )

    output = output_grouped.reshape(N, C_out, outH, outW)
    output += b[None, :, None, None]
    cache = (xp_grouped, kernel_grouped, pad, H, W, stride, num_groups)
    return output, cache


def conv2d_backward(dL_dout, cache):
    (xp_grouped, kernel_grouped, pad, H, W, stride, num_groups) = cache

    N, C_out, outH, outW = dL_dout.shape
    num_groups, C_out_per_group, C_in_per_group, kH, kW = kernel_grouped.shape
    dL_dout_grouped = dL_dout.reshape(N, num_groups, C_out_per_group, outH, outW)

    dL_dxp_grouped = np.zeros_like(xp_grouped)
    dL_dkernel_grouped = np.zeros_like(kernel_grouped)

    for h in range(dL_dout_grouped.shape[-2]):
        for w in range(dL_dout_grouped.shape[-1]):
            upstream = dL_dout_grouped[:, :, :, h, w]

            # Derivative with respect to input
            dL_dxp_grouped[
                :, :, :, stride * h : stride * h + kH, stride * w : stride * w + kW
            ] += np.einsum("ngo,goiyx->ngiyx", upstream, kernel_grouped)

            # Derivative with respect to weight
            patch = xp_grouped[
                :, :, :, stride * h : stride * h + kH, stride * w : stride * w + kW
            ]
            dL_dkernel_grouped += np.einsum("ngo,ngiyx->goiyx", upstream, patch)

    # Reshape
    dL_dkernel = dL_dkernel_grouped.reshape(C_out, C_in_per_group, kH, kW)
    dL_dxp = dL_dxp_grouped.reshape(N, -1, H + pad * 2, W + pad * 2)

    # Don't care about the derivative with respect to the padding
    dL_dx = dL_dxp[..., pad : pad + H, pad : pad + W]

    # Derivative with respect to bias
    dL_db = np.sum(dL_dout, axis=(0, 2, 3))
    return dL_dx, dL_dkernel, dL_db


def attn_forward(Q, K, V):
    """
    Q: (B, H, S, D)
    K: (B, H, T, D)
    V: (B, H, T, C)
    """
    B, H, S, D = Q.shape
    T = K.shape[2]

    # Queries are the rows, and Keys are the columns
    dots = np.einsum("bhsd,bhtd->bhst", Q, K)
    dots_scaled = dots / np.sqrt(D)  # B H S T

    exp_dots = np.exp(dots_scaled)  # B H S T
    attn_weights = exp_dots / np.sum(exp_dots, axis=-1, keepdims=True)

    out = np.einsum("bhst,bhtc->bhsc", attn_weights, V)
    cache = (Q, K, V, attn_weights)
    return out, cache


def attn_backward(dL_dout, cache):
    """
    dL_dout: B, H, S, C

    Out Step:
        out_sc = sum_t[attn_weights_st * V_tc]
        dout_sc / dattn_weights_st = V_tc
        dL / dattn_weights_st = sum_c[(dL / dout_sc) * (dout_sc / dattn_weights_st)]


    Backprop through softmax:
        attn_weights_t = exp_dots_t / sum_k[ exp_dots_k ]
        dattn_weights_t / dexp_dots_j = (f'g - g'f)/g**2

        Let g = sum_k[ exp_dots_k ]

        if t==j:
            (exp_dots_t * g - exp_dots_t * exp_dots_t)/(g**2)

        if t!=j:
            (- exp_dots_j * exp_dots_t)/(g**2)

        (delta(t==j) * exp_dots_t * g - exp_dots_j * exp_dots_t)/(g**2)

        (exp_dots_t / g)[delta(t==j) - exp_dots_j/g]

        attn_t[delta(t==j) - attn_j]

        dL / dexp_dots_j = sum_t(dL / dattn_weights_t * dattn_weights_t / dexp_dots_j)

        = sum_t(dL / dattn_weights_t * [attn_t[delta(t==j) - attn_j])

        = sum_t(dL / dattn_weights_t * [attn_t[- attn_j])
        + sum_t(dL / dattn_weights_t * attn_t * delta(t==j)]

        = sum_t(dL / dattn_weights_t * [attn_t[- attn_j])
        + (dL / dattn_weights_j * attn_j)]


        = - attn_j* sum_t(dL / dattn_weights_t * [attn_t])
        + (dL / dattn_weights_j * attn_j)]
    """

    (Q, K, V, attn_weights) = cache
    D = Q.shape[-1]

    # out = np.einsum("bhst,bhtc->bhsc", attn_weights, V)
    dL_attn_weights = np.einsum("bhsc,bhtc->bhst", dL_dout, V)
    dL_dV = np.einsum("bhsc,bhst->bhtc", dL_dout, attn_weights)

    # attn_weights = exp_dots / np.sum(exp_dots, axis=-1, keepdims=True)
    # exp_dots = np.exp(dots_scaled)  # B H S T
    # dots_scaled = dots / np.sqrt(D)  # B H S T
    dot = np.sum(dL_attn_weights * attn_weights, axis=-1, keepdims=True)
    dL_ddots = attn_weights * (dL_attn_weights - dot) * (1 / np.sqrt(D))

    # dots = np.einsum("bhsd,bhtd->bhst", Q, K)
    dL_dQ = np.einsum("bhst,bhtd->bhsd", dL_ddots, K)
    dL_dK = np.einsum("bhst,bhsd->bhtd", dL_ddots, Q)

    return dL_dQ, dL_dK, dL_dV


"""Activations"""


def relu_forward(x):
    out, cache = np.maximum(x, 0), (x,)
    return out, cache


def relu_backward(dL_dout, cache):
    """
    Since ReLU is elementwise
    We can think of this as a scalar function: derivative is 1 when x > 0, else 0.
    dL / dx_nj = dL / dout_nj * dout_nj / dx_nj
    dL / dout_nj * I(x > 0)
    """
    (x,) = cache
    dL_dx = (x > 0) * dL_dout
    return dL_dx


def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    cache = (out,)
    return out, cache


def sigmoid_backward(dL_dout, cache):
    """
    This operation is elementwise, so the backward is simple.
    1 / (1 + e^(-x))
    -1 / (1 + e^(-x))**2 * (-e^(-x))
    (e^(-x)) / (1 + e^(-x))**2
    [1 / (1 + e^(-x))] * [(e^(-x)) / (1 + e^(-x))]
    [1 / (1 + e^(-x))] * [(1 + e^(-x) - 1) / (1 + e^(-x))]
    [1 / (1 + e^(-x))] * [(1 + e^(-x) - 1) / (1 + e^(-x))]
    [1 / (1 + e^(-x))] * [1 - 1 / (1 + e^(-x))]
    """
    (out,) = cache
    return dL_dout * out * (1 - out)


def softmax_forward(x):
    N, K = x.shape
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    cache = (out,)
    return out, cache


def softmax_backward(dL_dout, cache):
    """
    Computing derivative with respect to one element in batch

    out_i = e^(x_i) / [e^(x_1) + ... + ... e^(x_k)]
    Quotient Rule: (f'g - g'f)/g^2
    dout_i / d_xi = (e^(x_i) g - e^(x_i) e^(x_i))/g^2
    = e^(x_i) / g - (e^(x_i) * e^(x_i)) / g^2
    (e^(x_i)/g) * (1 - e^(x_i))/g)

    dout_i / d_xj = (-g'f)/g^2
    = (-e^(x_j) *   e^(x_i)) / g^2
    = (e^(x_i)) / g) * (-e^(x_j)/g)

    dout_i / d_xj =
    = (e^(x_i)) / g) * (delta(i,j) -e^(x_j)/g)
    dL / d_xj
    = sum_i( (dL/dout_i) * (e^(x_i)) / g) * (delta(i,j) - e^(x_j)/g))
    = sum_i( (dL/dout_i) * (e^(x_i)) / g) * (- e^(x_j)/g))
        + sum_i( (dL/dout_i) * (e^(x_i)) / g) * (delta(i,j))
    =  (- e^(x_j)/g)) * sum_i ((dL/dout_i) * (e^(x_i)) / g))
        + (dL/dout_j) * (e^(x_j)) / g)
    = (e^(x_j)) / g) [dL/dout_j - sum_i ((dL/dout_i) * (e^(x_i)) / g))]
    """
    (out,) = cache
    N, K = out.shape
    dot = np.sum(dL_dout * out, axis=-1, keepdims=True)
    return out * (dL_dout - dot)


"""Normalization"""


def batchnorm_forward(x, g, b, eps=1e-5, bn_params=None):
    x_mean = np.mean(x, axis=0)
    x_demeaned = x - x_mean[None, :]
    x_var = np.mean(x_demeaned**2, axis=0)

    x_std = np.sqrt(x_var + eps)
    inv_std = 1 / x_std
    x_hat = x_demeaned * inv_std
    out = x_hat * g + b
    cache = (x_demeaned, x_var, x_std, inv_std, x_hat, g, eps)

    if bn_params is not None:
        bn_params["running_mean"] = bn_params["running_mean"] * bn_params[
            "momentum"
        ] + x_mean * (1 - bn_params["momentum"])
        bn_params["running_var"] = bn_params["running_var"] * bn_params[
            "momentum"
        ] + x_var * (1 - bn_params["momentum"])
    return out, cache


def batchnorm_backward(dL_dout, cache):
    """
    db:
        dout_ij / db_j = 1
        dL / db_j = sum_i (dL / dout_ij) * (dout_ij / b_j)
        = sum_i (dL / dout_ij)

    dg:
        out_ij = x_hat_ij * g_j + b_j
        dout_ij / dg_j = x_hat_ij
        dL / dg_j = sum_i (dL / dout_ij) * (dout_ij / g_j)
        = sum_i [(dL / dout_ij) * x_hat_ij]

    dxhat:
        dL / dx_hat_ij = (dL / dout_ij) * (dout_ij / x_hat_ij)
        = (dL / dout_ij) * g_j

    dinv_std:
        x_hat_ij = x_demeaned_ij * inv_std_j
        dx_hat_ij / dinv_std_j = x_demeaned_ij
        dL / dinv_std_j = sum_i(dL / dx_hat_ij * dx_hat_ij / dinv_std_j
        = sum_i (dL / dx_hat_ij * x_demeaned_ij)

    ddemeaned:
        dx_hat_ij / dx_demeaned_ij = inv_std_j
        dL / dx_demeaned_ij = dL / dx_hat_ij * dx_hat_ij / dx_demeaned_ij
        = (dL / dx_hat_ij) * inv_std_j

    std:
        inv_std_j = 1/std_j
        dL / dstd_j = dL / dinv_std_j * dinv_std_j / dstd_j
        = dL / dinv_std_j * (1 / std_j)**2

    var:
        std = np.sqrt(x_var + eps)
        dL / dx_var = dL / dstd * dstd / dx_var
        = dL / dstd * (1/(2*np.sqrt(x_var + eps)))

    ddmeaned:
        var_j = sum_i (x_demeaned_ij**2)/N
        var_j = (x_demeaned_1j**2 + ... + x_demeaned_nj**2) / N
        dvar_j / dx_demeaned_ij = 2 * x_demeaned_ij / N
        dL / dx_demeaned_ij = dL / dvar_j * dvar_j / dx_demeaned_ij
        = dL /dvar_j * (2 * x_demeaned_ij)/N

    dx:
        x_demeaned_ij = x_ij - (x_1j + ... + x_nj)/N
        dx_demeaned_ij / dx_kj = delta(k==i) - 1/N
        dL / dx_kj = sum_i [(dL / dx_demeaned_ij) * (dx_demeaned_ij / dx_kj)]
        dL / dx_kj = sum_i [(dL / dx_demeaned_ij) * (delta(k==i) - 1/N)]
        dL / dx_kj = sum_i [(dL / dx_demeaned_ij) * (-1/N)] + sum_i[(dL / dx_demeaned_ij) * delta(k==i)]
        dL / dx_kj = sum_i [(dL / dx_demeaned_ij) * (-1/N)] + (dL / dx_demeaned_kj)
    """
    x_demeaned, x_var, x_std, inv_std, x_hat, g, eps = cache
    N, D = x_demeaned.shape

    dL_db = np.sum(dL_dout, axis=0)
    dL_dg = np.sum(dL_dout * x_hat, axis=0)
    dL_dx_hat = dL_dout * g

    dL_dinv_std = np.sum(dL_dx_hat * x_demeaned, axis=0)
    dL_dx_demeaned = dL_dx_hat * inv_std

    dL_dx_std = dL_dinv_std * (-1 / x_std**2)
    dL_dx_var = dL_dx_std / (2 * np.sqrt(x_var + eps))
    dL_dx_demeaned += (2 * x_demeaned / N) * dL_dx_var

    dL_dx = -np.mean(dL_dx_demeaned, axis=0)[None] + dL_dx_demeaned
    return dL_dx, dL_dg, dL_db


def layernorm_forward(x, g, b, eps=1e-5):
    x_demeaned = x - np.mean(x, axis=1, keepdims=True)
    x_var = np.mean(x_demeaned**2, axis=1)
    x_std = np.sqrt(x_var + eps)
    inv_std = 1 / x_std

    x_hat = inv_std[:, None] * x_demeaned
    out = x_hat * g + b

    cache = (x_demeaned, x_var, x_std, inv_std, x_hat, g, eps)
    return out, cache


def layernorm_backward(dout, cache):
    """
    x_hat_ij = inv_std_i * x_demeaned_ij

    dx_hat_ij / dinv_std_i = x_demeaned_ij
    dL / dinv_std_i = sum_j( (dL / dx_hat_ij) * (dx_hat_ij / dinv_std_i)
    = sum_j( (dL / dx_hat_ij) * x_demeaned_ij)


    dx_hat_ij / x_demeaned_ij = inv_std_i
    dL / dx_demeaned_ij = (dL / dx_hat_ij) * (dx_hat_ij / dx_demeaned_ij)
    = (dL / dx_hat_ij) * inv_std_i



    dinv_std / dstd = -1/(std**2)


    Variance -> Demeaned
        var_i = sum_j(x_dm_i1**2 + ... + x_dm_in**2)/N
        dvar_i / dx_demeaned_ij = (2 * x_dm_ij)/N
        dL / dx_demeaned_ij = dL / dvar_i * dvar_i / dx_demeaned_ij

    Demeaned -> x
        x_demeaned_ij = x_ij - (x_i1 + ... + x_iN)/N
        dx_demeaned_ij / dx_ik = delta(j,k) - 1/N
        dL / dx_ik = sum_j(dL / dx_demeaned_ij * dx_demeaned_ij / dx_ik)
        = sum_j(dL / dx_demeaned_ij * [delta(j,k) - 1/N])
        = sum_j(dL / dx_demeaned_ij * delta(j,k)) - sum_j(dL / dx_demeaned_ij * [ 1/N])
        = dL / dx_demeaned_ij - sum_j(dL / dx_demeaned_ij * [1/N])

    """
    (x_demeaned, x_var, x_std, inv_std, x_hat, g, eps) = cache

    N, D = x_demeaned.shape

    db = np.sum(dout, axis=0)
    dg = np.sum(dout * x_hat, axis=0)
    dx_hat = dout * g

    dinv_std = np.sum(dx_hat * x_demeaned, axis=1)
    dx_demeaned = dx_hat * inv_std[:, None]

    dstd = dinv_std * (-1 / x_std**2)
    dvar = dstd * (1 / (2 * np.sqrt(x_var + eps)))

    dx_demeaned += dvar[:, None] * (2 * x_demeaned) / D  # Make sure this is D, not N
    dx = dx_demeaned - np.mean(dx_demeaned, axis=1, keepdims=True)

    return dx, dg, db


def rms_norm_forward(x, g, eps=1e-5):
    x_msv = np.mean(x**2, axis=1)
    x_rms = np.sqrt(x_msv + eps)
    inv_rms = 1 / x_rms

    x_hat = x * inv_rms[:, None]

    out = x_hat * g  # (,D)
    cache = (x, x_msv, x_rms, inv_rms, x_hat, g, eps)
    return out, cache


def rms_norm_backward(
    dL_dout: np.ndarray, cache: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """
    out_ij = x_hat_ij * g_j

    Derivative with respect to g_j
        dout_ij / dg_j = x_hat_ij
        dL / dg_j = sum_i[(dL / dout_ij) * (dout_ij / dg_j)]
        dL / dg_j = sum_i[(dL / dout_ij) * (x_hat_ij)]

    Derivative with respect to x_hat:
        dout_ij / dx_hat_ij = g_j
        dL / dx_hat_ij = (dL / dout_ij) * (dout_ij / dx_hat_ij)


    xhat_ij = x_ij * inv_std_i

    Derivative with respect to x:
        dxhat_ij / dx_ij = inv_std_i
        dL / dx_ij = (dL / dxhat_ij) * (dxhat_ij / dx_ij)
                   = (dL / dxhat_ij) * inv_std_i

    Derivative with respect to inv_std
        dxhat_ij / inv_std_i = x_ij
        dL / inv_std_i = sum_j[(dL / dxhat_ij) * (dxhat_ij / inv_std_i)]
                       = sum_j[(dL / dxhat_ij) * x_ij]

    Derivative with respect to std:
        inv_std = 1 / std
        dinv_std / dstd = -1 / std**2
        dL / std = (dL / dinv_std) * (dinv_std / dstd)

    Derivative with respect to the variance:
        x_std = sqrt(x_var + eps)
        dL / dx_var = dL / dx_std * dx_std / dvar_x
                    = (dL / dx_std) * 1/(2*sqrt(x_var + eps))

    Derivative with respect to the demeaned x:
        x_var_i = sum_j[x_demeaned_ij**2]/D
        dx_var_i / x_demeaned_ij = (2*x_demeaned_ij)/D

        dL / x_demeaned_ij = (dL / dx_var_i) * (dx_var_i / dx_demeaned_ij)
                           = (dL / dx_var_i) * (2*x_demeaned_ij)/D

    Derivative with respect to x (again):
        x_demeaned_ij = x_ij - sum_k[x_ik]/N

        dx_demeaned_ij / dx_ik = delta(k==j) - 1/N
        dL / dx_ik = sum_j[(dL / dx_demeaned_ij) * (dx_demeaned_ij / dx_ik)]
                   = sum_j[(dL / dx_demeaned_ij) * (delta(k==j) - 1/N)]
                   = sum_j[(dL / dx_demeaned_ij) * (delta(k==j)]
                     - sum_j[(dL / dx_demeaned_ij) * (1/N)]
                   = (dL / dx_demeaned_ik) - sum_j[(dL / dx_demeaned_ij) * (1/N)]
    """
    x, x_msv, x_rms, inv_rms, x_hat, g, eps = cache

    N, D = x.shape
    dL_dg = np.sum(dL_dout * x_hat, axis=0)  # (N,D) * (N,D) -> (D,)
    dL_xhat = dL_dout * g  # (N, D) * (D,) -> (N, D)

    dL_dx = dL_xhat * inv_rms[:, None]  # (N,D) * (N,1) -> (N,D)

    dL_inv_std = np.sum(dL_xhat * x, axis=1)  # (N,D) * (N,D)  -> (N,)

    dL_std = dL_inv_std * (-1 / x_rms**2)  # (N,)
    dL_dx_var = dL_std / (2 * np.sqrt(x_msv + eps))  # (N,)
    dL_dx += dL_dx_var[:, None] * (2 * x) / D  # (N,) * (N,D) -> (N,D)
    return dL_dx, dL_dg  # (N,D), (D,)


"""Losses"""


def mse_loss_forward(x, y):
    diff = x - y
    out = np.mean(diff**2)
    cache = (diff,)
    return out, cache


def mse_loss_backward(cache):
    """
    Note that dividing by N in the forward pass
    Divides all the gradients by N in the backwards -
    The sensitivity of output to input goes down by a factor of N
    """
    (diff,) = cache
    return 2 * diff / diff.size


def l1_loss_forward(x, y):
    diff = x - y
    out = np.mean(np.abs(diff))
    cache = (diff,)
    return out, cache


def l1_loss_backward(cache):
    (diff,) = cache
    return np.sign(diff) / diff.size


def softmax_loss_forward(x, y):
    N, K = x.shape
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    nll = -x[np.arange(N), y] + np.log(np.sum(exp_x, axis=-1))

    loss = np.mean(nll)
    cache = (probs, y)
    return loss, cache


def softmax_loss_backward(cache):
    """
    out = - x_y + log(e^x_1 + ...  + e^x_k)
    dout / dx_y = -1 + e^x_y/(e^x_1 + ...  + e^x_k)
    dout / dx_(j != y) = e^x_j/(e^x_1 + ...  + e^x_k)
    """
    (probs, y) = cache
    N, K = probs.shape
    dL_dx = probs
    dL_dx[np.arange(N), y] -= 1
    return dL_dx / N


def bce_loss_forward(p, y):
    """
    Assume p(y | x) is bernoulli, with parameter p = f(x)

    The data likelihood is
        prod_i [p(y_i | x_i)]

    The log likelihood is
        sum_i log(p(y_i | x_i))

    p(y_i | x_i) = p, if y_i = 1
    p(y_i | x_i) = (1-p), if y_i = 0

    sum_i_(y_i==1) [log(p)] + sum_i_(y_i==0) [log(1-p)]

    p = 1 / (1 + exp(-z))
    """
    likelihoods = y * np.log(p) + (1 - y) * np.log(1 - p)
    loss = -np.mean(likelihoods, axis=0)
    cache = (p, likelihoods, y)
    return loss, cache


def bce_loss_backward(cache):
    """
    L = -(l1 + ... + lN) / N
    (dL / dlikelihoods_i) = - 1 /N

    likelhoods_i = y_i * log(p) + (1 - y_i) * log(1-p)

    dlikelihoods_i / dp = y_i / p - (1 - y_i) / (1 - p)

    dL / dp_i = (dL / dlikelihoods_i) * (dlikelihoods_i / dp_is)
    """
    (p, likelihoods, y) = cache
    (N,) = p.shape
    dL_dlikelihoods = -(1 / N)
    dL_dp = dL_dlikelihoods * (y / p - (1 - y) / (1 - p))
    return dL_dp
