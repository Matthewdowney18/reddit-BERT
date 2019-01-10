import numpy as np
import torch
from torch.autograd import Variable
import os
import pickle
from torch.nn import Parameter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate import bleu_score


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking)

    lengths = masks.sum(dim=dim)

    return lengths


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
        #obj = obj
    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def argmax(inputs, dim=-1):
    values, indices = inputs.max(dim=dim)
    return indices


def get_sentence_from_indices(indices, vocab, eos_token, join=True):
    tokens = []
    for idx in indices:
        token = vocab.id2token[idx]

        if token == eos_token:
            break

        tokens.append(token)

    if join:
        tokens = ' '.join(tokens)

    tokens = tokens

    return tokens


def get_pretrained_embeddings(embeddings_dir):
    embeddings = np.load(embeddings_dir)
    emb_tensor = torch.FloatTensor(embeddings)
    return emb_tensor



def save_checkpoint(best_epoch, best_model, best_optimizer,
                    epoch, model, optimizer, train_loss, val_loss, metrics,
                    params, file):
    '''
    saves model into a state dict, along with its training statistics,
    and parameters
    :param best_epoch:
    :param best_model:
    :param best_optimizer:
    :param epoch:
    :param model:
    :param optimizer:
    :param train_loss:
    :param val_loss:
    :param metrics:
    :param params:
    :param file:
    :return:
    '''
    state = {
        'best_model': best_model.state_dict(),
        'best_optimizer': best_optimizer.state_dict(),
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_epoch': best_epoch,
        'epoch' : epoch,
        'train_loss' : train_loss,
        'val_loss' : val_loss,
        'metrics' : metrics,
        'params' : params,
        'file' : file,
        }
    torch.save(state, file["model_filename"])

def load_params(filename):
    '''
    loads the params of a saved model
    useful for when you want to load the model, and need to create a model
    with the same parameters first
    :param filename: filename of model
    :return: parameters (dict) files (dict)
    '''
    params = {}
    files = {}
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        params = checkpoint["params"]
        files = checkpoint["file"]
    else:
        print("no file found at {}".format(filename))
    return params, files


def load_checkpoint(filename, model, optimizer, use_best_model = True):
    '''
    loads previous model
    :param filename: file name of model
    :param model: model that contains same parameters of the one you are loading
    :param optimizer:
    :param use_best_model: true if you want to use the model with the lowest
        val loss
    :return: loaded model, checkpoint
    '''
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        if use_best_model:
            model.load_state_dict(checkpoint['best_model'])
            optimizer.load_state_dict(checkpoint['best_optimizer'])
        else:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def freeze_layer(layer, bool):
    '''
    freezes or unfreezes layer of pytorch model
    :param layer:
    :param bool: True to freeze, False to unfreeze
    :return:
    '''
    for param in layer.parameters():

        param.requires_grad = not bool
    layer.training = not bool
    return layer


def encoder_accuracy(targets, predicted):
    '''
    finds the token level, and sentence level accuracy
    :param targets: tensor of sentence
    :param predicted: tensor of predicted senences
    :return: sentence_accuracy, token_accuracy (float)
    '''
    batch_size = targets.size()
    sentence_acc = [1] * batch_size[0]
    token_acc = []
    for batch in range(0, batch_size[0]):
        for token in range(0, batch_size[1]):
            if targets[batch, token] != 0:
                if targets[batch, token] != predicted[batch, token]:
                    sentence_acc[batch] = 0
                    token_acc.append(0)
                else:
                    token_acc.append(1)

    sentence_accuracy = sum(sentence_acc)/len(sentence_acc)
    token_accuracy = sum(token_acc) / len(
        token_acc)
    return sentence_accuracy, token_accuracy


def add_negatve_class(outputs):
    '''

    :param outputs:
    :return:
    '''
    neg_prob = variable(torch.ones(outputs.size())) - outputs
    outputs = torch.cat((neg_prob, outputs), 1)
    return outputs


def classifier_accuracy(targets, predicted):
    '''
    calculates accuracy, precision, recall, and F1
    :param targets: numpy array of actual labels
    :param predicted: numpu array of predicted labels
    :return: accuracy, precision, recall, f1
    '''
    accuracy = accuracy_score(targets, predicted)
    precision = precision_score(targets, predicted)
    recall = recall_score(targets, predicted)
    f1 = f1_score(targets, predicted)
    return accuracy, precision, recall, f1



def load_features(filename):
    with open(filename,'rb') as f:
        features = pickle.load(f)
    return features

def save_features(filename, features):
    with open(filename, "wb") as f:
        pickle.dump(features, f)

