import torch.nn as nn
from transformers import BartForConditionalGeneration
import torch


class DocEncoder(nn.Module):
    def __init__(self, model_name='distilbart-xsum-12-6', device="cuda", **kwargs):
        super(DocEncoder, self).__init__()
        self.device = device

        gradient_checkpointing = False
        if self.training:
            gradient_checkpointing = True
            if torch.cuda.is_available():
                memory_in_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                if memory_in_gb > 40:
                    gradient_checkpointing = False

            print(f"Gradient Checkpointing: {gradient_checkpointing}\n")

        self.encoder = BartForConditionalGeneration.from_pretrained(
            model_name, output_hidden_states=False, gradient_checkpointing=gradient_checkpointing)

    def forward(self, example):
        output = self.encoder(**example, return_dict=False)
        if self.training:
            return output[0]
        else:
            return output
