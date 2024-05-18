import numpy as np
import torch
from loguru import logger
from codetiming import Timer

#元素级矩阵量化，根据指定的比特预算将输入矩阵进行量化
@Timer(name="quantize")
def quantize(
    X: torch.Tensor = None,
    B: int = 16,
    full_range: bool = False,
    simulate: bool = False,
    preserve_original_dtype=False,
    force=False
) -> torch.Tensor:
    """
    Element-wise matrix quantization for general bit-budget
    :param X (torch.Tensor): Matrix to be quantized
    :param B (int): Bit-budget per coordinate for quantization
    :param full_range (bool): If true, use bfloat16 when B=16, ignored otherwise
    :param simulate (bool): If true, simulate quantization using 64bits
    :param preserve_original_dtype (bool): If true, retains original dtype after quantization
    :return: Quantized matrix
    """
    orig_dtype = X.dtype
    device = X.device
    if B == 16 and device == torch.device("cpu"):
        logger.warning(
            f"Setting simulate = True as Half() dtype is not supported on CPU"
        )
        simulate = True
    
    match B:
        case 64 if not simulate:
            out = X.double()
        case 32 if not simulate:
            out = X.float()
        case 16 if not simulate:
            out = X.half() if not full_range else X.bfloat16()
        case _:
            if simulate:
                logger.warning(f"Forced quantization simulation to {B} bits")
            else:
                logger.warning(f"Using simulation to quantize to {B} bits")

            M = 2**B  # No. of quantization points per dimension
            res = 2 / (M - 1)  # Resolution

            # Normalize the coordinates of the quantizer input to [-1,1]
            X_min = X.min().item()
            X_max = X.max().item()
            # logger.trace(f"X_min = {X_min}, X_min.dtype = {X_min.dtype}, X_min.device = {X_min.device}")
            # logger.trace(f"X = {X}, X.dtype = {X.dtype}, X.device = {X.device}")
            X = torch.from_numpy(
                np.interp(X.to("cpu").numpy(), (X_min, X_max), (-1, 1))
            ).to(X.device)
            # logger.trace(f"X = {X}, X.dtype = {X.dtype}, X.device = {X.device}")

            # Quantize each coordinate with a scalar quantizer of unit dynamic range
            fst_pt = -1  # First quantization point
            L_idx = torch.floor((X - fst_pt) / res)  # Lower index for each entry
            #L、U是在量化点上的数值
            L = fst_pt + L_idx * res  # Matrix of lower quantization points
            U = fst_pt + (L_idx + 1) * res  # Matrix of upper quantization points

            # Nearest neighbor quantization 量化到最接近的量化点上
            Q = torch.zeros_like(X)
            Q[X < -1] = -1  # Value less than lower limit of dynamic range
            Q[X > 1] = 1  # Value more than upper limit of dynamic range
            #mask0、mask都是bool值 逻辑与运算之后表示那些值会被量化，就近量化
            mask0 = torch.abs(X) <= 1  # Value within dynamic range
            mask = torch.abs(X - L) <= res / 2
            Q[mask * mask0] = L[mask * mask0]
            mask = torch.abs(U - X) <= res / 2
            Q[mask * mask0] = U[mask * mask0]

            # Re-normalize the quantized matrix back to its input scale 将量化后的张量重新映射回原始输入数据的尺度
            Qr = torch.from_numpy(
                np.interp(
                    Q.to("cpu").numpy(),
                    (Q.min().item(), Q.max().item()),
                    (X_min, X_max),
                )
            ).to(X.device)
            logger.trace(f"Qr.dtype = {Qr.dtype}, Qr.device = {Qr.device}")
            out = Qr
    if preserve_original_dtype:
        out = out.to(orig_dtype)
    return out
