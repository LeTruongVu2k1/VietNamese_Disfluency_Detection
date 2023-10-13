from typing import List, Tuple
from collections import defaultdict
import torch
from datasets import Dataset
import numpy as np


from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from datasets import Features, Sequence, ClassLabel, Value

from customized_tokenizer import MyTokenizer

class SNR_Augmentation:
  """A Data-Augmentation Class using "Semantic-Neighbor-Replacement" method, based on this [article](https://aclanthology.org/2022.bionlp-1.12/)

  - First, we will get the entities' embeddings and group these entities with high cosine-similarity (the threshold for this cosine will be called ER_threshold (ER: Entity-replacement))
  - Then we will replace entities inside the sentence with their neighbor founds in step 1, the new sentence will be evaluated also by their embeddings' cosine-similarity  (the threshold for this cosine will be call SE_threshold (SE: Sentence-Evaluation)). Augmented sentences with cosine-similarity higher than that SE_threshold will be augmented to the original dataset.
  """
  def __init__(self, feature_extraction_tokenizer, feature_extraction_model, sentence_model, ER_threshold, SE_threshold, feature, device):
    self.device = device             # device for model to get embeddings
    self.feature_extraction_tokenizer = feature_extraction_tokenizer # HuggingFace's PretrainedTokenizer, must be initialized before
    self.feature_extraction_model = feature_extraction_model.to(self.device) # HuggingFace's Pretrained Model, must be initialized before
    self.sentence_model = sentence_model # sentence-transformers, must be initialized before

    self.ER_threshold = ER_threshold # Threshold of  cosine-similarity of two entity to consider if they're grouped inside a neighbor or not
    self.SE_threshold = SE_threshold # Threshold of cosine-similarity of two sentence (original and augmented sentence) to consider if they're augmented sentence are accepted or not
    self.feature = feature    # datasets.Features of input "dataset"

    self.RM_entities_embeddings = {} # keys: RM entities, values: entities' embedding
    self.IM_entities_embeddings = {} # keys: IM entities, values: entities' embedding

    self.RM_haved_neighbor = defaultdict(int)  # key: entity, value: neighbor-number of this entity
    self.IM_haved_neighbor = defaultdict(int)  # key: entity, value: neighbor-number of this entity

    self.RM_neighbors = defaultdict(list) # key: RM neighbor's number, value: List of entities have cosine-similarity above "ER_threshold"
    self.IM_neighbors = defaultdict(list) # key: IM neighbor's number, value: List of entities have cosine-similarity above "ER_threshold"

    self.RM_large_neighbor = {} # same structure with RM_neighbors but just store neighbors that have more than ONE entities
    self.IM_large_neighbor = {} # same structure with IM_neighbors but just store neighbors that have more than ONE entities


    self.augmented_data = {'text': [], 'label': []} # storing augmented data for loading into dataset later
    self.sentence_neighbor, self.label_neighbor = [], [] # storing pairs of original and augmented sentence (FOR VISUALIZING)
    self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) # Instantiate cosine-similarity's instance

  def __call__(self, dataset):
    self.get_all_embeddings(dataset) # get all entities' embeddings and store in self.RM_entities embeddings, self.IM_entities_embeddings

    self.RM_haved_neighbor, self.RM_neighbors = self.grouping_entity(self.RM_entities_embeddings, self.ER_threshold)
    self.IM_haved_neighbor, self.IM_neighbors = self.grouping_entity(self.IM_entities_embeddings, self.ER_threshold)

    # these 2 dictionaries will store neighbor numbers with more than one entities
    self.RM_large_neighbor = defaultdict(list, {k:v for k,v in self.RM_neighbors.items() if len(v) > 1})
    self.IM_large_neighbor = defaultdict(list, {k:v for k,v in self.IM_neighbors.items() if len(v) > 1})

    # sentence-replacement augmentation
    dataset.map(self.augmenting_each_sample)
    augmented_data = Dataset.from_dict(self.augmented_data, features=self.feature)

    return augmented_data

  def get_all_embeddings(self, dataset):
    dataset.map(self.get_embeddings_each_sample)

  # Function for getting word in text based on its offset
  def get_subword(self, text, offset_mapping, idx):
    start, end = offset_mapping[idx][0], offset_mapping[idx][1]
    return text[start:end]

  def get_embeddings_each_sample(self, example):
    tokenized_input = self.feature_extraction_tokenizer.tokenize_each_sample(example)

    not_tensor_type = ['labels', 'word_ids', 'offset_mapping']        # these values in "tokenized_input" is list-type and not needed for extracting embeddings
    to_cuda_input = {k: v.to(self.device) for k, v in tokenized_input.items() if k not in not_tensor_type} # use gpu for quick progressing
    feature_embeddings = self.feature_extraction_model(**to_cuda_input)['last_hidden_state'][0].detach().cpu() # get embeddings

    word_ids = tokenized_input['word_ids']
    aligned_labels = tokenized_input['labels']
    offset_mapping = tokenized_input['offset_mapping']

    i = 1
    while i < len(word_ids) - 1: # i need to be in range[1:n], because [None...None]
      if aligned_labels[i] in [0, 2]: # beginning tags (B-XX)
        s = aligned_labels[i]         # getting the tag (0: B-RM, 2: B-IM)
        embeddings = feature_embeddings[i]
        entity_name = self.get_subword(example['text'], offset_mapping, i) # use the entity names for keys in embedding's dictionary
        cnt = 1
        curr_word_ids = word_ids[i]

        i += 1
        while i < len(word_ids) and aligned_labels[i] == s + 1: # check if tag is I-XX
          embeddings += feature_embeddings[i]

          subword = self.get_subword(example['text'], offset_mapping, i)
          if curr_word_ids != word_ids[i]:
            entity_name += ' ' + subword
            curr_word_ids = word_ids[i]
          else:                           # Combining subword ... Ex: [n@@, ẵng] -> nẵng
            entity_name += subword

          cnt += 1
          i += 1

        if s == 0: # RM tags
          self.RM_entities_embeddings[entity_name] = embeddings / cnt
        else: # IM tags
          self.IM_entities_embeddings[entity_name] = embeddings / cnt

      else:
        i += 1

  def grouping_entity(self, entities_embeddings, threshold):

    keys = list(entities_embeddings.keys())    # list of entites
    haved_neighbor = defaultdict(int)          # each entity will have its neighbor number
    neighbors = defaultdict(list)              # each neighbor number will have a list of its neighbor entities

    # Instantiate cosine-similarity's instance
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    entity_threshold = threshold

    neighbor_num = 0 # neighbor number will be named from 1


    for i in range(len(keys)):
      if keys[i] in haved_neighbor: # check if this entity already grouped
        continue

      # if this entity is not grouped yet
      neighbor_num += 1
      haved_neighbor[keys[i]] = neighbor_num
      neighbors[neighbor_num].append(keys[i])

      for j in range(i+1, len(keys)):
        if keys[j] not in haved_neighbor: # check if this entity already grouped
          cosine_similarity = cos(entities_embeddings[keys[i]], entities_embeddings[keys[j]])

          if cosine_similarity >= entity_threshold:  # entity will be neighbor of each other if their cosine-similarity greater than entity_threshold
            neighbors[neighbor_num].append(keys[j])
            haved_neighbor[keys[j]] = neighbor_num

    return haved_neighbor, neighbors

  def get_entity_and_span(self, words, labels):
    """Slicing through "words", "labels" to get the span of entities
    Args:
      words: List of words
      labels: List of labels

    Return:
      spans: List of (start's index of entity, B-entity's tag, entity's name, entity's label)
    """
    spans = []

    i = 0

    while i < len(words):

      if labels[i] in [0, 2]: # 0: B-RM, 2: B-IM
        start = i # start of entity
        B_tag_value = labels[start]
        I_tag_value = B_tag_value + 1 # 1: I-RM, 3: I-IM

        i += 1
        while i < len(words) and labels[i] == I_tag_value:
          i += 1

        entity = words[start:i]         # modify array B which is created by sub-slicing array A wont't change A
        entity_label = labels[start:i]

        spans.append((start, B_tag_value, entity, entity_label))

      else:
        i += 1

    return spans




  # This function will replace entities with their neighbors in sequence.
  def sentence_replacement(self, sentence: List[str], label: List[str], spans: List[Tuple], possible_augmented_list: List):
    """Replace entities with their neighbors and append them to "possible_augmented_list". This function will replace entity inside "spans" recursively.
    Args:
      sentence: List of words
      label: List of labels
      spans: List of entity found in "sentence"
      span_idx: the index of element inside "spans"
      debt: the diffence of length between the neighbor-replaced sentence and its original. For the first entity of "spans", debt will be 0.
        But in the 2nd, 3rd,... entity, when adding these entities, we need to consider "debt" because we are slicing using the index of original
        sentence but the length between the original and augmented one is not equal.
      possible_augmented_list: List of (augmented_sentence, augmented_label)
    """
    queue = []

    debt = 0 # for the first loop, debt remain 0
    queue.append((sentence, label, debt))

    for span_idx in range(len(spans)):

      start = spans[span_idx][0] # start index of entity
      tag_type = spans[span_idx][1] # 0: RM - 1: IM

      entity = spans[span_idx][2]       # entity
      entity_label = spans[span_idx][3] # label of entity

      neighbor_num = self.RM_haved_neighbor[' '.join(entity)] if tag_type == 0 else self.IM_haved_neighbor[' '.join(entity)]
      neighbors = self.RM_large_neighbor[neighbor_num] if tag_type == 0 else self.IM_large_neighbor[neighbor_num]

      if neighbors == []: continue # skip if this entity have no neighbors

      for i in range(len(queue)):
        sentence, label, debt = queue.pop(0)

        for neighbor in neighbors:
          if neighbor != ' '.join(entity):
            if tag_type == 0:
              neighbor_label = [0] + [1 for _ in range(len(neighbor.split(' ')) - 1)]
            else:
              neighbor_label = [2] + [3 for _ in range(len(neighbor.split(' ')) - 1)]

            neighbor = neighbor.split()  # because `sentence` is list-type
            augmented_sentence = sentence[0:(start + debt)] + neighbor + sentence[(start + debt + len(entity)):]
            augmented_label = label[0:(start + debt)] + neighbor_label + label[(start + debt + len(entity)):]

            new_debt = debt + (len(neighbor) - len(entity))

            possible_augmented_list.append((augmented_sentence, augmented_label))

            queue.append((augmented_sentence, augmented_label, new_debt))




  def compare_sentence_embeddings_and_augment(self, possible_augmented_list, original_sentence, original_label, threshold):
    ############ Getting Sentence Embeddings ############
    for possible_augmented_text, possible_augmented_label in possible_augmented_list:
      augmented_sentence_embeddings = self.sentence_model.encode(' '.join(possible_augmented_text))
      original_sentence_embeddings = self.sentence_model.encode(' '.join(original_sentence))

      # print(' '.join(possible_augmented_text))

      # Calculate the dot product of the two arrays
      dot_product = np.dot(augmented_sentence_embeddings, original_sentence_embeddings)

      # Calculate the Euclidean norm for both vectors
      norm1 = np.linalg.norm(augmented_sentence_embeddings)
      norm2 = np.linalg.norm(original_sentence_embeddings)

      # Calculate the cosine similarity
      cosine_similarity = dot_product / (norm1 * norm2)


      if (cosine_similarity > threshold) and cosine_similarity < 1:
        self.augmented_data['text'].append(' '.join(possible_augmented_text))
        self.augmented_data['label'].append(possible_augmented_label)

        self.sentence_neighbor.append([' '.join(original_sentence), ' '.join(possible_augmented_text)])
        self.label_neighbor.append([original_label, possible_augmented_label])

    # still adding original text and label if there is no entity's neighbor
    self.augmented_data['text'].append(' '.join(original_sentence))
    self.augmented_data['label'].append(original_label)




  def augmenting_each_sample(self, example):
    spans = []  # values: (start, tag_type, entity, entity_label)

    words = example['text'].split()
    labels = example['label']         # Note: this variable is a list

    spans = self.get_entity_and_span(words, labels) # getting span of each entity inside the sentence

    ### Getting possible sentences ###
    possible_augmented_list = [] # storing text and label which is possible for augmented
    self.sentence_replacement(sentence=words, label=labels, spans=spans, possible_augmented_list=possible_augmented_list)

    self.compare_sentence_embeddings_and_augment(possible_augmented_list, words, labels, self.SE_threshold)



def data_augment(dataset, pretrained_feature_extraction_checkpoint, pretrained_sentence_extraction_checkpoint, class_names, ER_threshold, SE_threshold):
    feature_extraction_model = AutoModel.from_pretrained(pretrained_feature_extraction_checkpoint)

    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_feature_extraction_checkpoint)
    feature_extraction_tokenizer = MyTokenizer(pretrained_tokenizer=pretrained_tokenizer, padding=True, truncation=True, return_tensors='pt')

    sentence_model = SentenceTransformer(pretrained_sentence_extraction_checkpoint)

    # class_names = ['B-RM', 'I-RM', 'B-IM', 'I-IM', 'O']
    features = Features({'text': Value('string'), 'label': Sequence(ClassLabel(names=class_names))})

    customed_augmentation = SNR_Augmentation(feature_extraction_tokenizer=feature_extraction_tokenizer,
                                             feature_extraction_model=feature_extraction_model,
                                             sentence_model=sentence_model,
                                             ER_threshold=ER_threshold,
                                             SE_threshold=SE_threshold,
                                             feature=features,
                                             device='cuda')
    
    augmented_dataset = customed_augmentation(dataset)

    return augmented_dataset


