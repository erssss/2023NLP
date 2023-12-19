# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

def Metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="macro")
    target_names = ["class_0", "class_1"]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=3)

    print(
        "Accuracy: {:.1%}\nPrecision: {:.1%}\nRecall: {:.1%}\nF1: {:.1%}".format(
            accuracy, macro_precision, macro_recall, weighted_f1
        )
    )
    print("classification_report:\n")
    print(report)


def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (
        batch_seqs,
        batch_seq_masks,
        batch_seq_segments,
        batch_labels,
    ) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        seqs, masks, segments, labels = (
            batch_seqs.to(device),
            batch_seq_masks.to(device),
            batch_seq_segments.to(device),
            batch_labels.to(device),
        )
        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}".format(
            batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1)
        )
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader):
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels in dataloader:
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return (
        epoch_time,
        epoch_loss,
        epoch_accuracy,
        roc_auc_score(all_labels, all_prob),
        all_prob,
    )


def test(model, dataloader):
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels in dataloader:
            batch_start = time.time()
            seqs, masks, segments, labels = (
                batch_seqs.to(device),
                batch_seq_masks.to(device),
                batch_seq_segments.to(device),
                batch_labels.to(device),
            )
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= len(dataloader.dataset)

    return batch_time, total_time, accuracy, all_prob
