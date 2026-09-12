import torch


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, G+k, out_dim)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        
    '''
    
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jkl->ijl', b_splines, coef.to(b_splines.device))
    
    return y_eval


def _generic_local_basis(values, spans, grid, k):
    """Compact Cox--de Boor recurrence for non-cubic/adaptive cases."""
    in_dim, batch = values.shape
    grid_size = grid.shape[1]
    offsets = torch.arange(2 * k + 1, device=values.device)
    candidates = spans[:, :, None] - k + offsets[None, None, :]
    valid_span = (spans >= 0) & (spans < grid_size - 1)
    basis = (
        (offsets[None, None, :] == k) & valid_span[:, :, None]
    ).to(dtype=values.dtype)
    knot_offsets = torch.arange(2 * k + 2, device=values.device)
    knot_indices = spans[:, :, None] - k + knot_offsets[None, None, :]
    valid_knots = (knot_indices >= 0) & (knot_indices < grid_size)
    expanded_grid = grid[:, None, :].expand(in_dim, batch, grid_size)
    gathered_knots = torch.gather(
        expanded_grid, 2, knot_indices.clamp(0, grid_size - 1)
    )
    u = values[:, :, None]
    for degree in range(1, k + 1):
        count = 2 * k + 1 - degree
        # Out-of-support gathers clamp multiple invalid knots to an endpoint.
        # Masking after 0/0 hides forward NaNs but not their backward gradients.
        # These terms are zero in the native global recurrence; make their
        # denominators safe before division, then discard the invalid terms.
        left_denominator = (
            gathered_knots[:, :, degree : degree + count]
            - gathered_knots[:, :, :count]
        )
        right_denominator = (
            gathered_knots[:, :, degree + 1 : degree + 1 + count]
            - gathered_knots[:, :, 1 : count + 1]
        )
        left = (
            (u - gathered_knots[:, :, :count])
            / torch.where(left_denominator != 0, left_denominator, 1.0)
            * basis[:, :, :count]
        )
        right = (
            (gathered_knots[:, :, degree + 1 : degree + 1 + count] - u)
            / torch.where(right_denominator != 0, right_denominator, 1.0)
            * basis[:, :, 1 : count + 1]
        )
        valid = (
            valid_knots[:, :, :count]
            & valid_knots[:, :, degree : degree + count]
            & valid_knots[:, :, 1 : count + 1]
            & valid_knots[:, :, degree + 1 : degree + 1 + count]
        )
        basis = torch.where(valid, left + right, 0.0)
    return basis, candidates[:, :, : k + 1]


def coef2curve_local(x_eval, grid, coef, k, uniform_grid=False):
    """Evaluate only the ``k + 1`` B-splines active at each sample."""
    batch, in_dim = x_eval.shape
    grid_size = grid.shape[1]
    coefficient_count = grid_size - k - 1
    values_transposed = x_eval.transpose(0, 1)
    if uniform_grid:
        spacing = (grid[:, 1] - grid[:, 0])[:, None]
        spans = torch.floor(
            (values_transposed - grid[:, 0, None]) / spacing
        ).to(dtype=torch.long)
    else:
        spans = torch.searchsorted(
            grid.contiguous(), values_transposed.contiguous(), right=True
        ) - 1

    basis, coefficient_indices = _generic_local_basis(
        values_transposed, spans, grid, k
    )
    valid_coefficients = (
        (coefficient_indices >= 0)
        & (coefficient_indices < coefficient_count)
    )
    coefficient_indices = coefficient_indices.clamp(0, coefficient_count - 1)
    output_dim = coef.shape[2]
    coefficients_by_basis = coef.reshape(
        in_dim * coefficient_count, output_dim
    )
    input_offsets = (
        torch.arange(in_dim, device=x_eval.device)[:, None, None]
        * coefficient_count
    )
    flat_indices = (coefficient_indices + input_offsets).reshape(-1)
    selected = coefficients_by_basis.index_select(0, flat_indices).reshape(
        in_dim, batch, k + 1, output_dim
    )
    selected = selected * valid_coefficients[:, :, :, None]
    return torch.sum(basis[:, :, :, None] * selected, dim=2).permute(1, 0, 2)


def curve2coef(x_eval, y_eval, grid, k):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
        lamb : float
            regularized least square lambda
            
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, G+k, out_dim)
    '''
    #print('haha', x_eval.shape, y_eval.shape, grid.shape)
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
    #print('mat', mat.shape)
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
    #print('y_eval', y_eval.shape)
    device = mat.device
    
    #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
    except:
        print('lstsq failed')
    
    # manual psuedo-inverse
    '''lamb=1e-8
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:,:,:,0]'''
    
    # Store the frequently traversed output dimension contiguously.  Upstream
    # PyKAN returns [input, output, basis], which forces the numerical fast
    # path to repair the layout on every evaluation.
    return coef.permute(0, 2, 1).contiguous()


def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid
