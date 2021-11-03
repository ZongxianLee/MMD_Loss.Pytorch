import torch
import torch.nn as nn



def broadcast_values(x,y):
	"""
	Utility function that make two tensors broadcastable
	Necessary for computing the kernel
	"""
	length_x = len(x.shape)
	length_y = len(y.shape)
	#if we consider the 1D dimension kernel 
	if x.shape[-1]!= y.shape[-1]:
		#reshaping for broadcasting
		x_scaled = x.view(*x.shape,*([1]*length_y))
		y_scaled = y.view(*([1]*length_x),*y.shape)
		
	# if we have multiple dimensions
	elif length_x>1 and length_y>1:
		x_scaled = x.view(*x.shape[:-1],*([1]*(length_y-1)),x.shape[-1])
		y_scaled = y.view(*([1]*(length_x-1)),*y.shape)

	#case x is 1D
	elif length_x==1 and length_y>1:
		x_scaled = x.view(*([1]*(length_y-1)),x.shape[0])
		y_scaled = y

	#case y is 1D
	elif length_y==1 and length_x>1:
		x_scaled = x
		y_scaled = y.view(*([1]*(length_x-1)),y.shape[0])

	#case are both 1D
	else:
		x_scaled=x
		y_scaled=y


	return x_scaled,y_scaled

class GaussianKernel():
	def __init__(self,sigmas):
		"""
		Simple version of the gaussian kernel with 
		"""
		self.sigmas_square = sigmas**2
	def __call__(self,x,y=None):
		if y is None:
			y = x
		x,y = broadcast_values(x,y)
		L2 = (x-y)**2/self.sigmas_square
		return torch.exp(torch.sum(-L2,dim=-1))

class MMDLoss(nn.Module):
	def __init__(self,kernel):
		super(MMDLoss,self).__init__()
		"""
		Function to compute the MMD loss on two samples of data
		It only works for discrete distributions
		Parameters:
			kernel: a callable kernel object
		"""
		self.kernel = kernel

	def forward(self,x,y):
		complete = torch.cat([x,y],dim=0)
		kernel_complete = self.kernel(complete)
		size_x = x.shape[0]
		kernel_x = kernel_complete[:size_x,:size_x]
		kernel_y = kernel_complete[size_x:,size_x:]

		kernel_xy = kernel_complete[:size_x,size_x:]
		kernel_yx = kernel_complete[size_x:,:size_x]


		return torch.mean(kernel_x) + torch.mean(kernel_y) -torch.mean(kernel_xy) - torch.mean(kernel_yx)



if __name__ == "__main__":
	x = torch.randn(1000,4)
	y = torch.randn(10000,4)
	gaussiankernel = GaussianKernel(torch.ones(4)*2)
	loss_fn = MMDLoss(gaussiankernel)
	print(loss_fn(x,y))
