# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os

location = os.environ.get("LOCATION")

class BertModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.model_name_or_path = os.path.join(location, "model", "bert-base-uncased")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path, 
            num_labels=self.num_labels, 
            output_attentions=False,
            # torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", 
            output_hidden_states=False,
            # attn_implementation="eager"
            # torch_dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def forward(self, input_id, mask, label):
        output = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output

class AlBertModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(AlBertModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.model_name_or_path = os.path.join(location, "model", "albert")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path, 
            num_labels=self.num_labels, 
            output_attentions=False,
            # torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", 
            output_hidden_states=False,
            # attn_implementation="eager"
            # torch_dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def forward(self, input_id, mask, label):
        output = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output

class GPTModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(GPTModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.model_name_or_path = os.path.join(location, "model", "gpt")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False,
            # torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_id, mask, label):
        output = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
