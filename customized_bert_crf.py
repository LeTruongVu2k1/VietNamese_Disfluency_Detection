from torchcrf import CRF
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import PreTrainedModel


class BERTCRF(PreTrainedModel):
  def __init__(self, num_labels, id2label, label2id,
               encoder_checkpoint, hidden_dropout, attention_dropout):
    encoder = AutoModel.from_pretrained(encoder_checkpoint, hidden_dropout_prob=hidden_dropout, attention_probs_dropout_prob=attention_dropout)
    config = encoder.config

    super(BERTCRF, self).__init__(config)

    self.encoder = encoder
    self.config = config

    self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
    self.classifier = nn.Linear(self.config.hidden_size, num_labels)
    self.crf = CRF(num_tags=num_labels+1, batch_first=True)

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      labels=None
  ):
      """
      labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
          Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
          1]``.
      """
      outputs = self.encoder(
          input_ids,
          attention_mask=attention_mask,
      )

      sequence_output = outputs.last_hidden_state
      sequence_output = self.dropout(sequence_output)
      logits = self.classifier(sequence_output)

      # dropping bos_tag (input_ids=2) and eos_tag (input_ids=0)
      condition = torch.logical_not(torch.isin(input_ids, torch.tensor([2,0]).to('cuda')))

      crf_logits = logits[condition].reshape(logits.shape[0], -1, logits.shape[2]) # reshape needed due to boolean indexing of 2D-tensor will return 1D-tensor
      padding_logits = torch.zeros(crf_logits.shape[0], crf_logits.shape[1], 1).to('cuda')
      crf_logits = torch.cat((crf_logits, padding_logits), dim=-1)
      crf_labels = labels[condition].reshape(labels.shape[0], -1)
      crf_labels = torch.where(crf_labels == -100, 5, crf_labels) # we will consider padding token (label=-100) as a new label, so
                                                                  # from 0 to 4 is our current label, and padding as a new label with value 5
      crf_attention_mask = attention_mask[:,2:].byte() # because attention_mask just contain 0 and 1, deleting bos and eos's postions same for deleting first two element
                                                       # could be replaced by: crf_attention_mask = crf_attention_mask[condition].reshape(crf_attention_mask.shape[0], -1)


      loss = None
      if labels is not None:
          log_likelihood, tags = (self.crf(crf_logits, crf_labels, crf_attention_mask),
                                  self.crf.decode(crf_logits, crf_attention_mask))
          loss = 0 - log_likelihood
          return {'loss': loss, 'logits': crf_logits}
      else:
          tags = self.crf.decode(crf_logits, crf_attention_mask)
          return {'logits': crf_logits}

      # tags = torch.Tensor(tags)

      # output = (tags,) + outputs[2:]
      # return ((loss,) + output) if loss is not None else output