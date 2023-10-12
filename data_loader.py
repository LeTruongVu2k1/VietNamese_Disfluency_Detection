import os
from git import Repo
import pandas as pd
from datasets import load_dataset
from datasets import Features, Sequence, ClassLabel, Value
import argparse

# clone dataset's repository
def clone_repo(url, repo_dir):
    git_url = url
    repo_dir = repo_dir

    Repo.clone_from(git_url, repo_dir)


# create dataset directories for storing .pkl files later
def make_dir(root_dir='data', split_dirs=['train', 'test', 'split']):
    os.mkdir(root_dir)

    for split in split_dirs:
        os.mkdir(os.path.join(root_dir, split))


# reading .in, .out files in pd.DataFrame, merging text and label together,
# then export them as pkl. files
def load_to_pkl(file_path, dst_path, file_names=['seq.in', 'seq.out']):
  text_path = os.path.join(file_path, file_names[0]) # path to 'seq.in'
  label_path = os.path.join(file_path, file_names[1]) # path to 'seq.out'

  text_df = pd.read_csv(text_path, header=None, sep='\t')  
  label_df = pd.read_csv(label_path, header=None, sep='\t')

  if text_df.shape != label_df.shape:
    print("Number of sentences and labels not aligned!")
    return

  full_df = pd.concat([text_df, label_df], axis=1)
  full_df.columns = ['text', 'label']
  full_df['label'] = full_df['label'].map(lambda x: x.split()) # convert label from string to list


  '''
    Note: Cause this [problem](https://stackoverflow.com/questions/23111990/pandas-dataframe-stored-list-as-string-how-to-convert-back-to-list) happend with `label` column, I have to export them as `.pkl` file instead fof `.csv`
  '''
  full_df.to_pickle(f'{dst_path}/data.pkl')

  # if dont' use index=False, it will automatically add 'index' as new column
  # full_df.to_p(f'{dst_path}/data.csv', index=False) 




def load_data(data_dir):
  """ 
      Loading "datasets.dataset" from "data_dir" - which contain '.pkl' files. 

      Args:
        data_dir: The directory to load dataset.

      Return:
        A single HuggingFace's datasets.Dataset variable contains both train, dev, test.
  """

  '''
  In `load_dataset`, the loading script I used is *pandas pickled dataframe (with the `pandas` script)* 
  (see more in [local file section](https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html))
  '''
  data_dict = {k: f'{data_dir}/{k}/data.pkl' for k in ['train', 'dev', 'test']}

  class_names = ['B-RM', 'I-RM', 'B-IM', 'I-IM', 'O']
  features = Features({'text': Value('string'), 'label': Sequence(ClassLabel(names=class_names))})

  dataset = load_dataset('pandas', data_files=data_dict, features=features) # using `pandas pickled dataframe`
  '''
  We can load dataset using `datasets.Dataset.from_pandas` method. 
  However, we can only load in 1 split each time, so IMO it's better to load all 3 splits into one in-memory-variable  because for my personal experience.
  
  from datasets import Dataset

  features = Features({'text': Value('string'), 'label': Sequence(ClassLabel(names=class_names))})

  dataset = Dataset.from_pandas(full_df, split='train', features=features)
  '''

  return dataset   

if __name__ == '__main__':
    ''' If you run this data_loader.py, it will progress as the list below: 
    - Reading seq.in, seq.out as pd.DataFrame, then combining them as one pd.DataFrame
    - Converting pd.DataFrame to .pkl file, storing them in 'data/[train, test, split]' directories
    
    Note: Use "load_data" to load these .pkl files, storing them in a SINGLGE variable `dataset`
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-git_url', type=str, default='https://github.com/VinAIResearch/PhoDisfluency.git', help="Repository of Vietnamese Disfluency Dataset to clone")

    parser.add_argument('-repo_dir', type=str, default='PhoDisfluency', help="Place to store the above repo after cloning")


    parser.add_argument('-src_parent_dir', type=str, default="PhoDisfluency/data/word-level", help="Parent Directory which store the Original Vietnamese Disfluency Dataset")

    parser.add_argument('-dst_parent_dir', type=str, default="data", help="Destination Directory which store the `.pkl` files for loading into datsets.Dataset")


    args = parser.parse_args()

    # cloning Vietnamese Disfluency Dataset and store in "repo_dir"
    clone_repo(url=args.git_url, repo_dir=args.repo_dir)

    # create directory for storing .pkl files
    root_dir = args.dst_parent_dir
    os.mkdir(root_dir)

    split_dirs = ["train", "test", "dev"]
    for split in split_dirs:
      os.mkdir(os.path.join(root_dir, split))

    # reading .in, .out files in Vietnamese Disfluency Dataset origin repository and export them to .pkl files 
    splits = ["train", "test", "dev"]

    for split in splits:
        src_path = os.path.join(args.src_parent_dir, split)
        dst_path = os.path.join(args.dst_parent_dir, split)

        load_to_pkl(src_path, dst_path) 



    



