import evaluate
import torch

metric = evaluate.load("seqeval")

import numpy as np

# def compute_metrics_crf(eval_preds):
#     logits, labels = eval_preds
#     labels = labels[:,1:-1] # ignore bos_tag and eos_tag
#     predictions = model.crf.decode(torch.from_numpy(logits).to('cuda'))

#     assert len(labels[0]) == len(predictions[0])

#     # Remove ignored index (special tokens) and convert to labels
#     true_labels = [[label_names[l] for l in label if l not in [5, -100]] for label in labels]
#     true_predictions = [
#         [label_names[p] for (p, l) in zip(prediction, label) if l not in [5, -100]]

#         for prediction, label in zip(predictions, labels)
#     ]
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }

def compute_metrics_crf_with_extra(label_names, model):
    def compute_metrics_crf(eval_preds):
        logits, labels = eval_preds
        labels = labels[:,1:-1] # ignore bos_tag and eos_tag
        predictions = model.crf.decode(torch.from_numpy(logits).to('cuda'))

        assert len(labels[0]) == len(predictions[0])

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l not in [5, -100]] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l not in [5, -100]]

            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
    return compute_metrics_crf

# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)

#     # Remove ignored index (special tokens) and convert to labels
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_predictions = [
#         [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }


def compute_metrics_with_extra(label_names):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    return compute_metrics

