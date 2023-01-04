from inspect import Attribute
import torch
from torch import nn, Tensor

from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
     def __init__(self, model:nn.Module, layers : Iterable[str]):
          super().__init__()
          self.model = model
          self.layers = layers
          # self._features = {layer: torch.empty(0) for layer in layers} 
          self.model.eval()
          for layer_id in layers:
               # print("layer id : ",layer_id)
               layer = dict([*self.model.named_modules()])[layer_id]
               layer.register_forward_pre_hook(self.save_outputs_hook(layer_id))
               
     def save_outputs_hook(self, layer_id = str) -> Callable:          
          def fn(_, input) :
               # pass
               if len(input[0].shape) == 4 : 
                    input[0][:] = torch.round(input[0])
                    # print("extract input! :", input[0].shape)
                    # print("extract output! :", output.shape)
                    # self._features[layer_id] = output
          return fn
     
     def forward(self, x:Tensor) :
          return self.model(x)




          
          
