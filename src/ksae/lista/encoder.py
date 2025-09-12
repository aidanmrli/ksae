from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from .shrinkage import soft_threshold


class LISTA(nn.Module):
	"""
	Learned Iterative Shrinkage-Thresholding Algorithm encoder.

	Computes an approximate sparse code Z for input X using a fixed number of
	iterations with learned parameters (W_e, S, theta).
	"""

	def __init__(
		self,
		input_dim: int,
		code_dim: int,
		num_iterations: int = 5,
		learn_tied: bool = True,
		init_scale: float = 1.0,
	):
		super().__init__()
		self.input_dim = input_dim
		self.code_dim = code_dim
		self.num_iterations = num_iterations
		self.learn_tied = learn_tied

		# Parameters
		self.We = nn.Parameter(torch.empty(code_dim, input_dim))
		self.S = nn.Parameter(torch.empty(code_dim, code_dim))
		self.theta = nn.Parameter(torch.full((code_dim,), 0.1))

		self.reset_parameters(init_scale)

	def reset_parameters(self, init_scale: float = 1.0) -> None:
		# Kaiming uniform for We; small init for S near identity interaction
		nn.init.kaiming_uniform_(self.We, a=5 ** 0.5)
		with torch.no_grad():
			self.S.copy_(torch.eye(self.code_dim))
			self.S.mul_(0.0)
			self.theta.clamp_(min=1e-3)
			self.theta.mul_(init_scale)

	def forward(self, X: Tensor, return_all: bool = False) -> Tensor | Tuple[Tensor, dict]:
		"""
		X: shape (batch, input_dim) or (input_dim,)
		Returns Z of shape (batch, code_dim) or (code_dim,) matching batch dims of X.
		"""
		is_vector = X.dim() == 1
		if is_vector:
			X = X.unsqueeze(0)

		B = X @ self.We.t()  # (batch, code_dim)
		Z = soft_threshold(B, self.theta)

		if return_all:
			history = {
				"B": B,
				"C": [],
				"Z": [Z],
			}

		# Perform exactly T iterations as per LISTA spec (t = 1..T)
		for _ in range(self.num_iterations):
			C_t = B + Z @ self.S.t()
			Z = soft_threshold(C_t, self.theta)
			if return_all:
				history["C"].append(C_t)
				history["Z"].append(Z)

		if is_vector:
			Z = Z.squeeze(0)
			if return_all:
				for k in ("B",):
					history[k] = history[k].squeeze(0)
				history["Z"] = [z.squeeze(0) for z in history["Z"]]
				history["C"] = [c.squeeze(0) for c in history["C"]]

		return (Z, history) if return_all else Z


