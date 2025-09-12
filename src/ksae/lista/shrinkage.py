import torch
from torch import Tensor


def soft_threshold(values: Tensor, thresholds: Tensor) -> Tensor:
	"""
	Component-wise soft-thresholding (shrinkage) operator.

	For each element i: sign(v_i) * max(|v_i| - theta_i, 0).
	Broadcasts thresholds across batch and features as needed.
	"""
	if thresholds.ndim == 1:
		# allow [m] thresholds broadcast over batch
		while thresholds.ndim < values.ndim:
			thresholds = thresholds.unsqueeze(0)
		thresholds = thresholds.expand_as(values)
	return torch.sign(values) * torch.relu(torch.abs(values) - thresholds)


def soft_threshold_backward_mask(values: Tensor, thresholds: Tensor) -> Tensor:
	"""
	Returns a mask for the derivative of soft-thresholding wrt input values.

	Subgradient: 1 where |v_i| > theta_i, else 0. Non-differentiable at equality.
	Useful if implementing custom backward; PyTorch autograd usually suffices.
	"""
	if thresholds.ndim == 1:
		while thresholds.ndim < values.ndim:
			thresholds = thresholds.unsqueeze(0)
		thresholds = thresholds.expand_as(values)
	return (torch.abs(values) > thresholds).to(values.dtype)


