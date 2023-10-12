from data_augmentation import SNR_Augmentation
from customized_tokenizer import MyTokenizer
from data_loader import load_data
import argparse


from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from datasets import Features, Sequence, ClassLabel, Value

from data_augmentation import data_augment

from transformers import DataCollatorForTokenClassification

from utils import compute_metrics_with_extra, compute_metrics_crf_with_extra

from transformers import AutoModelForTokenClassification

from transformers import TrainingArguments

from transformers import Trainer
import transformers

from customized_bert_crf import BERTCRF

def train(args, dataset): 
    """ Training with Data-Augmentation """

    # List of label's names (string) 
    label_names = dataset['train'].features['label'].feature.names
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_feature_extraction_checkpoint)

    customized_tokenizer = MyTokenizer(pretrained_tokenizer=tokenizer, padding=True, truncation=True)

    dataset_v2 = customized_tokenizer(dataset, remove_columns=dataset["train"].column_names)
    
    if args.training_version in ['DA', 'BOTH']:
        # Data augmentation
        augmented_dataset = data_augment(dataset['dev'], args.pretrained_feature_extraction_checkpoint, args.pretrained_sentence_extraction_checkpoint, label_names, args.ER_threshold, args.SE_threshold)

        # print(f"Grouped {len(augmented_dataset.RM_large_neighbor)} RM-entities, {len(augmented_dataset.IM_large_neighbor)} IM-entities.")
        print(f"Size of Augmented Dataset: {len(augmented_dataset)}")

        dataset_v3_train = customized_tokenizer(augmented_dataset, remove_columns=dataset['train'].column_names)
    else: 
        # CRF version
        dataset_v3_train = dataset_v2['train']
    
    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training 
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    if args.training_version == 'DA':
        # In this version, we will use pretrained phoBERT from Huggingface Transformers Library.
        model = AutoModelForTokenClassification.from_pretrained(
            args.pretrained_feature_extraction_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )

        compute_metrics = compute_metrics_with_extra(label_names)

    else:
        model = BERTCRF(num_labels=len(label_names),
                id2label=id2label,
                label2id=label2id,
                encoder_checkpoint=args.pretrained_feature_extraction_checkpoint,
                hidden_dropout=args.hidden_dropout,
                attention_dropout=args.attention_dropout)
        
        compute_metrics = compute_metrics_crf_with_extra(label_names, model)
        
        

        
    # train_arguments = TrainingArguments(
    #                 output_dir="DD0101/modular_augmentation",
    #                 evaluation_strategy="epoch",
    #                 save_strategy="epoch",
    #                 save_total_limit=3,
    #                 learning_rate=5e-5,
    #                 num_train_epochs=2,
    #                 push_to_hub=False,
    #                 per_device_train_batch_size=32,
    #                 per_device_eval_batch_size=32,
    #                 metric_for_best_model='eval_f1', # These are set for
    #                 load_best_model_at_end=True      # early stopping call backs
    #             )

    

    train_arguments = TrainingArguments(
                output_dir=args.output_dir,
                evaluation_strategy=args.evaluation_strategy,
                save_strategy=args.save_strategy,
                save_total_limit=args.save_total_limit,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_train_epochs,
                push_to_hub=args.push_to_hub,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                metric_for_best_model=args.metric_for_best_model,       # These are set for
                load_best_model_at_end=args.load_best_model_at_end      # early stopping call backs
            )
    

    trainer = Trainer(
        model=model,
        args=train_arguments,
        train_dataset=dataset_v3_train,
        eval_dataset=dataset_v2["dev"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=25)]
    )
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Parameters in Hugging Face's TrainingArguments. More information please visit "https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments"
    parser.add_argument('-data_dir', type=str, default='data', help="The directory to load dataset, which contain s .pkl files")

    parser.add_argument('-training_version', type=str, choices=['DA', 'CRF', 'BOTH'], default='DA', help='DA: Data-Augmentation, CRF: CRF-plugin model, BOTH: Both DA and CRF')

    parser.add_argument('-output_dir', type=str, default='DD0101', help="The output directory where the model predictions and checkpoints will be written")    

    parser.add_argument('-evaluation_strategy', type=str, choices=['epoch', 'steps', 'no'], default='epoch', help="The evaluation strategy to adopt during training")    

    parser.add_argument('-save_strategy', type=str, choices=['epoch', 'steps', 'no'], default='epoch', help="The checkpoint strategy to adopt during training")  

    parser.add_argument('-save_total_limit', type=int, default=3, help="If a value is passed, will limit the total amount of checkpoints")

    parser.add_argument('-learning_rate', type=float, default=5e-5, help="The initial learning rate for AdamW optimizer")

    parser.add_argument('-num_train_epochs', type=int, default=3, help="Total number of training epochs to perform")

    parser.add_argument('-push_to_hub', type=bool, default=False, help="Whether or not to push the model to the Hub every time the model is saved")

    parser.add_argument('-per_device_train_batch_size', type=int, default=32, help="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training")

    parser.add_argument('-per_device_eval_batch_size', type=int, default=32, help="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation")

    parser.add_argument('-metric_for_best_model', type=str, default='eval_f1', help="Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models") 

    parser.add_argument('-load_best_model_at_end', type=bool, default='True', help="Whether or not to load the best model found during training at the end of training") 

    # Paramameter using in 'DA' training version (Data-Agumentation)
    parser.add_argument('-pretrained_tokenizer_checkpoint', type=str, default='vinai/phobert-base', help="Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models")  

    parser.add_argument('-ER_threshold', type=float, default=0.8, help="Threshold of cosine-similarity of two entity's embeddings to consider if these two entities are grouped or not")

    parser.add_argument('-SE_threshold', type=float, default=0.85, help="Threshold of cosine-similarity of two sentences's embeddings (augmented vs original) to consider if this augmented sentence is chosen or not")

    parser.add_argument('-pretrained_feature_extraction_checkpoint', type=str, default='vinai/phobert-base', help="Checkpoint for Huggingface's PretrainedTokenizer and PretrainedModel")  

    parser.add_argument('-pretrained_sentence_extraction_checkpoint', type=str, default='sentence-transformers/paraphrase-MiniLM-L6-v2', help="Checkpoint for pretrained's sentence transformers to extract sentence's embedding")  

    # Paramameter using in 'CRF' or 'BOTH' training version 
    parser.add_argument('-hidden_dropout', type=float, default=0.2, help="Hidden-dropout's probability in BERT")
    parser.add_argument('-attention_dropout', type=float, default=0.4, help="Attention-dropout's probability in BERT")
 
    args = parser.parse_args()

    dataset = load_data(args.data_dir)

    train(args, dataset)



    
    