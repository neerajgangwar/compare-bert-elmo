import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from allennlp.modules.scalar_mix import ScalarMix


class BCN(nn.Module):
    """implementation of Biattentive Classification Network in 
    Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    for text classification"""
    def __init__(self, config, bilm):
        super().__init__()
        self.emb_size = config['emb_size']
        self.fc_hidden_size = config['fc_hidden_size']
        self.bilstm_encoder_size = config['bilstm_encoder_size']
        self.bilstm_integrator_size = config['bilstm_integrator_size']
        self.fc_hidden_size1 = config['fc_hidden_size1']
        self.mem_size = config['mem_size']
        self.bilm = bilm
        self.scalar_mix = ScalarMix(mixture_size=self.bilm.n_layers, do_layer_norm=False)

        self.fc = nn.Linear(self.emb_size, self.fc_hidden_size)

        self.bilstm_encoder =  nn.LSTM(
            input_size=self.fc_hidden_size,
            hidden_size=int(self.bilstm_encoder_size/2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=config['dropout']
        )

        self.bilstm_integrator = nn.LSTM(
            input_size=self.bilstm_encoder_size * 3,
            hidden_size=int(self.bilstm_integrator_size/2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=config['dropout']
        )

        self.attentive_pooling_proj = nn.Linear(self.bilstm_integrator_size, 1)

        self.fc1 = nn.Linear(self.bilstm_integrator_size * 4, self.fc_hidden_size1)
        self.fc2 = nn.Linear(self.fc_hidden_size1, self.mem_size)

        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        self.log_sm = nn.LogSoftmax()
        self.dropout = nn.Dropout(config['dropout'])

        self.device = config['device']


    def makeMask(self, lens, hidden_size):
        mask = []
        max_len = max(lens)
        for l in lens:
            mask.append([1]*l + [0]*(max_len - l))
        mask = Variable(torch.FloatTensor(mask))
        if hidden_size == 1:
            trans_mask = mask
        else:
            trans_mask = mask.unsqueeze(2).expand(mask.size(0), mask.size(1), hidden_size)
        
        return trans_mask.to(self.device)


    def forward(self, input, input_len):
        """
        input: batch_size x seq_len
        input_len: batch_size
        """
        batch_size = input.size(0)
        _, hidden = self.bilm(input, input_len)
        emb = self.scalar_mix(hidden)
        reps = self.dropout(emb)

        max_len = max(input_len)
        input_len = input_len.cpu().numpy()

        compressed_reps = reps.view(-1, self.emb_size)
        task_specific_reps = self.relu(self.fc(compressed_reps)).view(batch_size, max_len, self.fc_hidden_size)
        task_specific_reps = pack(task_specific_reps, input_len, batch_first=True, enforce_sorted=False)
        
        outputs, _ = self.bilstm_encoder(task_specific_reps)
        X, _ = unpack(outputs, batch_first=True)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = X.bmm(X.permute(0, 2, 1).contiguous())

        attention_mask1 = Variable((-1e7 * (attention_logits <= 1e-7).float()).data)
        masked_attention_logits = attention_logits + attention_mask1
        compressed_Ay = self.sm(masked_attention_logits.view(-1, max_len))
        attention_mask2 = Variable((attention_logits >= 1e-7).float().data) # mask those all zeros
        Ay = compressed_Ay.view(batch_size, max_len, max_len) * attention_mask2

        Cy = torch.bmm(Ay, X) # batch_size * max_len * bilstm_encoder_size

        # Build the input to the integrator
        integrator_input = torch.cat([Cy, X - Cy, X * Cy], 2)
        integrator_input = pack(integrator_input, input_len, batch_first=True, enforce_sorted=False)

        outputs, _ = self.bilstm_integrator(integrator_input) # batch_size * max_len * bilstm_integrator_size
        Xy, _ = unpack(outputs, batch_first=True)

        # Simple Pooling layers
        max_masked_Xy = Xy + -1e7 * (1 - self.makeMask(input_len, self.bilstm_integrator_size))
        max_pool = torch.max(max_masked_Xy, 1)[0]
        min_masked_Xy = Xy + 1e7 * (1 - self.makeMask(input_len, self.bilstm_integrator_size))
        min_pool = torch.min(min_masked_Xy, 1)[0]
        mean_pool = torch.sum(Xy, 1) / torch.sum(self.makeMask(input_len, 1), 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.attentive_pooling_proj(Xy.contiguous().view(-1, self.bilstm_integrator_size))
        self_attentive_logits = self_attentive_logits.view(batch_size, max_len) + -1e7 * (1 - self.makeMask(input_len, 1))
        self_weights = self.sm(self_attentive_logits)
        self_attentive_pool = torch.bmm(self_weights.view(batch_size, 1, max_len), Xy).squeeze(1)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self.dropout(pooled_representations)

        rep = self.dropout(self.relu(self.fc1(pooled_representations_dropped)))
        rep = self.dropout(self.relu(self.fc2(rep)))

        return rep, None


class Classifier(nn.Module):
    def __init__(self, config, classes):
        super().__init__()

        self.classes = classes
        self.input_size = config['mem_size']
        self.hidden_sizes = [int(x) for x in  config['hid_sizes_cls'].split(',') if x]
        assert len(self.hidden_sizes) >= 1
        self.layers = len(self.hidden_sizes) + 1 # including the output layer

        self.dropout = nn.Dropout(config['dropout'])

        self.mlps = nn.ModuleList()
        for i in range(self.layers):
            if i == 0:
                self.mlps.append(nn.Linear(self.input_size, self.hidden_sizes[i]))
            elif i < self.layers - 1:
                self.mlps.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
        self.mlps.append(nn.Linear(self.hidden_sizes[-1], self.classes))

        self.tanh = nn.Tanh()
        # self.log_sm = nn.LogSoftmax()


    def forward(self, rep):
        '''AutoModelForSequenceClassification
        rep: batch x mem_size
        '''
        for i in range(self.layers-1):
            rep = self.dropout(self.tanh(self.mlps[i](rep)))
        logit = self.mlps[self.layers-1](rep)

        # output_sm = self.log_sm(logit)
        return logit


class RepModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self, config, num_classes, bilm):
        super().__init__()

        self.encoder = BCN(config, bilm)

        self.classifier = Classifier(config, num_classes)

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, input, input_len):
        outputs = self.encoder(input, input_len)
        sentence_emb = outputs[0]
        extra_out = outputs[1]

        logits = self.classifier(sentence_emb)

        return logits, extra_out # extra_out can be 'attention weight' or other intermediate information
