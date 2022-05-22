import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# ====================================================
# Model
# ====================================================
class DebertaBaseLastMLMLastFourLayer(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained("microsoft/deberta-base", output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained("microsoft/deberta-base", config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
            
        self.fc_dropout1_1 = nn.ModuleList([nn.Dropout(cfg.fc_dropout) for _ in range(cfg.dropout_num)])
        self.fc1_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size//4)
        self.fc_dropout1_2 = nn.ModuleList([nn.Dropout(cfg.fc_dropout) for _ in range(cfg.dropout_num)])
        self.fc1_2 = nn.Linear(self.config.hidden_size, self.config.hidden_size//4)
        self.fc_dropout1_3 = nn.ModuleList([nn.Dropout(cfg.fc_dropout) for _ in range(cfg.dropout_num)])
        self.fc1_3 = nn.Linear(self.config.hidden_size, self.config.hidden_size//4)
        self.fc_dropout1_4 = nn.ModuleList([nn.Dropout(cfg.fc_dropout) for _ in range(cfg.dropout_num)])
        self.fc1_4 = nn.Linear(self.config.hidden_size, self.config.hidden_size//4)
        
        self.fc_dropout2 = nn.ModuleList([nn.Dropout(cfg.fc_dropout) for _ in range(cfg.dropout_num)])
        self.fc2 = nn.Linear((self.config.hidden_size//4)*4, 128)
        
        self.fc_dropout3 = nn.Dropout(cfg.fc_dropout)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        
        self._init_weights(self.fc1_1)
        self._init_weights(self.fc1_2)
        self._init_weights(self.fc1_3)
        self._init_weights(self.fc1_4)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def apply_dropout(self, nn_linear, dropouts, inputs):
        rets = [nn_linear(dropout(inputs)).unsqueeze(-1) for dropout in dropouts]
        return torch.cat(rets, axis=-1).mean(axis=-1)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        
        last_hidden_states = [
            self.apply_dropout(self.fc1_1, self.fc_dropout1_1, outputs[1][-1]).unsqueeze(-1),
            self.apply_dropout(self.fc1_2, self.fc_dropout1_2, outputs[1][-2]).unsqueeze(-1),
            self.apply_dropout(self.fc1_3, self.fc_dropout1_3, outputs[1][-3]).unsqueeze(-1),
            self.apply_dropout(self.fc1_4, self.fc_dropout1_4, outputs[1][-4]).unsqueeze(-1)
        ]
        last_hidden_states = self.relu(torch.cat(last_hidden_states,axis = -1).mean(axis=-1))
        
        last_hidden_states = self.relu(self.apply_dropout(self.fc2, self.fc_dropout2, last_hidden_states))

        output = self.fc3(self.fc_dropout3(last_hidden_states))
        return output