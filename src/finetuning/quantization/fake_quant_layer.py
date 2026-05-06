import torch
import torch.nn as nn

class FakeQuantLinear(nn.Module):
    def __init__(self, 
                 linear: nn.Linear, 
                 bits: int = 4) -> None:
        super().__init__()
        
        '''
        Description:
            A fake-quantization wrapper around nn.Linear that simulates INT-N weight quantization
            during the forward pass while preserving gradient flow via the Straight-Through Estimator (STE).
            
            During forward: weights are quantized (round + clamp) then dequantized back to floating point.
            During backward: gradients pass through as if quantization didn't happen (STE).
            
            This allows the model to learn weights that are robust to quantization noise,
            without actually reducing precision during training.

        Args:
            linear (nn.Linear): The original linear layer to wrap with fake quantization.
            bits (int): Number of quantization bits. Default: 4 (INT4, range [-8, 7]).

        Note:
            - Do NOT access weights via `.data` — it detaches from the computation graph
            and kills STE gradient flow (see failure_analysis #15).
            - The wrapped linear's weight remains the trainable parameter. FakeQuantLinear
            does not create new parameters — it reuses the original nn.Linear.
        '''
        
        self.linear = linear
        self.bits = bits
        self.register_buffer('scale', torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor
            ) -> torch.Tensor:
        
        '''
        Description:
            Forward pass with simulated INT-N quantization on weights using symmetric
            per-channel (per-output-row) quantization scheme and STE for gradient flow.

        Args:
            x (torch.Tensor): Input activation tensor of shape (batch, seq_len, in_features).

        Returns:
            torch.Tensor: Output of shape (batch, seq_len, out_features), computed as
                F.linear(x, w_fake, bias) where w_fake uses quantized weights in the
                forward path but routes gradients through the original weights.

        Note:
            - Scale is computed per output channel: scale_i = max(|W_i|) / (2^(bits-1) - 1)
            - STE trick: w_fake = quant(W).detach() + W - W.detach()
              Forward sees quantized weights; backward sees identity gradient to W.
        '''
        
        # W = self.linear.weight.data   # ← .data detaches from computation graph!
        W = self.linear.weight
        scale = torch.max(W.abs(), dim = 1)[0] / (2 ** (self.bits - 1) - 1)
        scale = scale.clamp(min = 1e-8) # Prevent division by zero
        W_q = torch.clamp( (W / scale[:, None]).round(), -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1)
        
        '''
        The fake quantization is implemented by detaching the quantized weights from the computation 
        graph and adding the difference between the original weights and the detached quantized weights 
        back to the original weights. This allows the gradients to flow through the original weights 
        during backpropagation, while still using the quantized weights for the forward pass.
        '''
        w_fake = (W_q * scale[:, None]).detach() + W - W.detach()
        
        # Use the quantized weights for the forward pass, but keep the original weights for backpropagation
        return nn.functional.linear(x, w_fake, self.linear.bias) 