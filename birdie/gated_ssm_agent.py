import torch

from torch.optim import Adam, AdamW
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

softmax = nn.Softmax()
sigmoid = nn.Sigmoid()


class RMSNorm(nn.Module):
	def __init__(self, dims, scale_init_fn=torch.ones, eps=1e-8):
		super(RMSNorm, self).__init__()
		self.eps = eps
		self.scale = nn.Parameter(scale_init_fn(dims))

	def forward(self, x):

		norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
		return self.scale * x / (norm + self.eps)



class RMS_split(nn.Module):
	'''
		This normalizes the training losses and actions separately.
	'''
	def __init__(self, input_dim, output_dim=None, dropout_rate=0.0,):
		super(RMS_split, self).__init__()

		self.ls_norm = RMSNorm(dims=input_dim//2, )

		self.rs_norm = RMSNorm(dims=input_dim//2)
		self.dropout = nn.Dropout(dropout_rate)

		self.scale = nn.Parameter(torch.zeros(input_dim)+0.1)

		self.output_dim = output_dim
		if self.output_dim is not None:
			self.out = nn.Linear(input_dim, output_dim, bias=False,)

	def forward(self, x):
		x_left, x_right = torch.chunk(x, 2, dim=-1)
		x_left = self.ls_norm(x_left)
		x_right = self.rs_norm(x_right)
		x = torch.cat([x_left, x_right], dim=-1)
		x = x * self.scale
		if self.output_dim is None:
			return x
		return self.out(x)
		# return self.dropout(self.out(x))
	


class SwiGLU(nn.Module):
	def __init__(self, dims, hidden_dims):
		super(SwiGLU, self).__init__()
		self.input_dims = dims
		self.hidden_dims = hidden_dims

		self.wi = nn.Linear(dims, hidden_dims*2, bias=False,)
		self.wo = nn.Linear(hidden_dims, dims, bias=False,)

	def forward(self, x):
		residual = x
		x = self.wi(x)
		x, gate = x.chunk(2, dim=-1)
		x = x * F.sigmoid(gate)
		return self.wo(x) + residual

class InPlaceSwiGLU(nn.Module):
	def __init__(self, in_dims, hidden_dims, out_dims, bias=False,):
		super(InPlaceSwiGLU, self).__init__()
		self.input_dims = in_dims
		self.hidden_dims = hidden_dims

		self.wi = nn.Linear(in_dims, hidden_dims*2, bias=bias,)
		self.wo = nn.Linear(hidden_dims, out_dims, bias=bias,)

	def forward(self, x):
		x = self.wi(x)
		x, gate = x.chunk(2, dim=-1)
		x = x * F.sigmoid(gate)
		return self.wo(x)

	


class GatedSSM(nn.Module):
	def __init__(self, dims, state_size_mult=2, dropout_rate=0.0, ):
		super(GatedSSM, self).__init__()

		# self.norm = RMSNorm(dims)
		self.norm = RMS_split(dims)
		state_dims = (dims * state_size_mult)
		self.K_proj = nn.Linear(dims, state_dims, bias=False,)

		self.ugg_proj = nn.Linear(dims, state_dims*3, bias=False,)

		self.out_proj = nn.Linear(state_dims, dims, bias=False,)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, x, ):
		residual = x
		x = self.norm(x)

		K = self.K_proj(x)
		K = F.sigmoid(K)

		(u, g_in, g_out) = self.ugg_proj(x).chunk(3, dim=-1)
		u = u * F.sigmoid(g_in) * (1-K)
		g_out = F.sigmoid(g_out)

		outputs = [
			u[..., 0:1, :]
		]
		for idx in range(1, K.shape[-2]):
			A0 = K[..., idx-1:idx, :]
			B0 = outputs[-1]
			B1 = u[..., idx:idx+1, :]
			outputs.append(A0 * B0 + B1)
		outputs = torch.concat(outputs, dim=-2)
		outputs = outputs * g_out
		y = self.out_proj(outputs)

		return y + residual


class MLPModel(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dims=[64, 64,], dropout_rate=0.0, num_heads=1,):
		super(MLPModel, self).__init__()
		self.layers = nn.ModuleList()

		# Reprojects the input to the hidden dimensions after normalization.
		self.layers.append(RMS_split(input_dim, hidden_dims[0], dropout_rate=dropout_rate,))

		for i in range(len(hidden_dims)):
			self.layers.append(GatedSSM(dims=hidden_dims[-1], dropout_rate=dropout_rate,))

		self.layers.append(RMS_split(hidden_dims[-1]))
		self.layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=False,))

		self.network = nn.Sequential(*self.layers)

	def forward(self, x, attention_mask=None,):
		output = self.network(x)
		if self.eval():
			output = output.cpu()
		return output