class MyTokenizer:
  """
  The vinai/phobert-base is a slow-tokenizer, we are not able to accesss features like "word_ids", "offset_mapping".
  This MyTokenizer is create to mimic fast-tokenizer in Huggingface, the preprocessed dataset
  returned has attributes like "word_ids", "offset_mapping".

  PLEASE NOTE THAT THIS IS ONLY FOR GETTING "word_ids", "offset_mapping", THIS CLASS IS NOT INHERIT FROM ANY HUGGINGFACE'S PRETRAINED-TOKENIZER.

  !!!CAUTION: If "return_tensors" == 'pt', it will return 2D torch tensor instead of 1D! Use 'pt' if you want to put your tensor to the model
  (model requires 2D tensor).
  """
  def __init__(self, pretrained_tokenizer, padding, truncation, return_tensors=None):
    '''
    pretrained_tokenizer: the HuggingFace's pretrained tokenizer, you must initialize it first and pass here
    padding: boolean, the padding strategy tokenizing your input using "pretrained_tokenizer"
    truncation: boolean, the truncation strategy tokenizing your input using "pretrained_tokenizer"
    return_tensors: optional, tensor returned after tokenizing your input using "pretrained_tokenizer"
    '''
    self.pretrained_tokenizer = pretrained_tokenizer
    self.padding = padding
    self.truncation = truncation
    self.return_tensors = return_tensors

  def __call__(self, dataset, remove_columns=[]):
    # Mapping your entire dataset with "tokenize_each_sample"
    self.remove_columns = remove_columns
    return dataset.map(self.tokenize_each_sample, remove_columns=remove_columns)


  def tokenize_each_sample(self, example):
    """Used for mapping the tokenizing function to each of sample of the `datasets.Dataset` input.
    Args:
      example: each sample of the original `datasets.Dataset`, must contain 'text' and 'label' field

    Return:
      Tokenized dataset
    """
    tokens = self.pretrained_tokenizer.tokenize(example['text'])

    word_ids, offset_mapping = self.token2word_ids_and_offset_mapping(tokens)

    aligned_labels = self.align_labels_with_tokens(example['label'], word_ids)

    tokenized_input = self.pretrained_tokenizer(example['text'], padding=self.padding, truncation=self.truncation, return_tensors=self.return_tensors)

    tokenized_input['labels'] = aligned_labels

    tokenized_input['word_ids'] = word_ids

    tokenized_input['offset_mapping'] = offset_mapping


    return tokenized_input



  def token2word_ids_and_offset_mapping(self, token):
    """Create word's index and offset mappings based on corresponding list of tokens
    Arg:
        token: List of tokens

        Note: List of tokens does not contain eos or bos tag. We can tokenize a text by using tokenizer.tokenize().
              This frustration due to the misalignment between the word-indexes created by tokenizer.tokenize(text) vs tokenizer(index).
              After a few experiences, I found that using tokenizer.tokenize() is more accurated.
              Unlike tokenizer(text) that tokenizer.tokenize(text) doesn't include bos or eos.

      Return:
        List of word-index and offset-mapping correspondings to the tokens.
    """
    word_ids = [None]
    word_id = 0

    seek = 0
    offset_mapping_list = [(0, 0)]

    for item in token:
      word_ids.append(word_id)

      if item[-2:] == '@@':
        offset_mapping_list.append((seek, seek + len(item) - 2))
        seek += len(item) - 2

      else:
        word_id += 1

        offset_mapping_list.append((seek, seek + len(item)))
        seek += len(item) + 1

    offset_mapping_list.append((0, 0))

    word_ids.append(None)

    return word_ids, offset_mapping_list


  def align_labels_with_tokens(self, labels, word_ids):
    """Align label to the tokens based on their word's index
    Args:
      labels: a list of labels
      word_ids: a list of word index

    Return:
      new_label: list of new labels aligned with their corresponding word's index
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 0 and label != 4:
                label += 1
            new_labels.append(label)

    return new_labels

  def display_aligned_tokens_labels(self, token, word_ids, aligned_labels, label_names):
    """Use this function for display token, word-index, aligned labels of each "sample inside datasets.Dataset after mapping tokenize_function".
    Args:
      token: List of token
      word_ids: List of word's index
      aligned_labels: List of aligned labels
      label_names: List of labels' name

    Usage example:
      If your dataset after mapping `tokenize_function` is for example `dataset_v2`, use can display the first sample of `dataset_v2` like this:

        train_ds = dataset_v2['train']
        tokens = tokenizer.convert_ids_to_tokens(train_ds[0]['input_ids'][0])
        display_aligned_tokens_labels(tokens, train_ds[0]['word_ids'][0], train_ds[0]['labels'][0])

    """

    print('Token'.ljust(20) + 'Word_ids'.ljust(20) + 'Aligned Label'.ljust(20) + 'Label Names')
    print()
    for i in range(len(token)):
      print(token[i].ljust(20) + str(word_ids[i]).ljust(20) + str(aligned_labels[i]).ljust(20) + str(label_names[aligned_labels[i]] if aligned_labels[i] != -100 else aligned_labels[i]))


   