import torch
import torch.nn as nn

from collections import OrderedDict

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size, optimizer_class, optimizer_kwargs, max_grad_norm):
        """_summary_

        Args:
            input_size (_type_): _description_
            output_size (_type_): _description_
            max_grad_norm (float): max norm of the gradients for gradient clipping
                                    to prevent gradient explosion
        """
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.max_grad_norm = max_grad_norm

        # For compatibility with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def build_optimizer(self, optimizer_class, optimizer_kwargs):
        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_soize`, and returns the updated parameters of the neural 
        network.
        """
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

        nn.utils.clip_grad_norm_(updated_params.values(), self.max_grad_norm)

        # self.optimizer.zero_grad()
        # loss.backward(retain_graph = True)
        # #Clip grad norm
        # nn.utils.clip_grad_norm_(params.values(), self.max_grad_norm)
        # self.optimizer.step()

        # return OrderedDict(self.named_meta_parameters())
        return updated_params
