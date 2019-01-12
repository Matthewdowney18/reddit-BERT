import numpy as np
import torch
import torch.utils.data
import time
import os
import argparse
from tqdm import tqdm
import json

from pytorch_pretrained_bert.modeling import BertForNextSentencePrediction
from pytorch_pretrained_bert.optimization import BertAdam

from classification.dataset import make_dataloader
from classification.utils import variable, cuda, classifier_accuracy, \
     load_checkpoint, load_results, load_features, load_args, \
     save_checkpoint


def main():
    parser = argparse.ArgumentParser()

    #files arguments
    parser.add_argument("-model_name", default="classifacation_2",
                        help="Name of the model. To be used in filename")
    parser.add_argument("-model_group", default="baseline",
                        help="dir to put model in")
    parser.add_argument("-old_model_name", default=None,
                        help="filename of model to be loaded")
    parser.add_argument("-model_version", default= 0,
                        help="version to be added to the models filename")
    parser.add_argument("-bert_model", default="bert-base-uncased",
                        help="Bert pre-trained model")
    parser.add_argument("-train", default=True,
                        help="true if you want to train the model")
    parser.add_argument("-val", default=True,
                        help="true if you want to evaluate the model")
    parser.add_argument("-train_batch_size", default=65,
                        help="batch size for training dataset")
    parser.add_argument("-val_batch_size", default=10,
                        help="batch size for validation dataset")
    parser.add_argument("-max_len", default=60,
                        help="max length for input sequence")
    parser.add_argument("-learning_rate", default=0.00005,
                        help="learning weight for optimization")
    parser.add_argument("-gradient_accumulation_steps", default=1,
                        help="")
    parser.add_argument("-num_epochs", default=2,
                        help="number of epochs for training, and validation")
    parser.add_argument("-num_outpout_checkpoints_train", default=-2,
                        help="determines how many times the loss is outputed "
                        "during training")
    parser.add_argument("-num_outpout_checkpoints_val", default=1,
                        help="determines how many times metrics are outputed "
                        "during training")
    parser.add_argument("-warmup_proportion", default=0.1,
                        help="Proportion of training to perform linear learning"
                        " rate warmup for.")


    args = parser.parse_args()
    args.project_file = os.getcwd()
    args.dataset_path = "{}/data/".format(args.project_file)
    args.output_dir = '{}/classification/{}/{}_{}/'.format(args.project_file,
        args.model_group, args.model_name, args.model_version)

    os.makedirs(args.output_dir, exist_ok=True)

    # check_files(file)

    use_old_model = args.old_model_name is not None
    if use_old_model:
        args.old_model_filename = '{}/classification/{}/{}' \
        '/'.format(args.project_file,
            args.model_group, args.old_model_name)


        args = load_args("{}args.json".format(args.old_model_filename), args)

    model = cuda(BertForNextSentencePrediction.from_pretrained(
        args.bert_model))

    phases = []
    data_loaders = []

    #loads features from file that is created in make_features.py
    #it takes a long time to create features which is why there is a seperate
    #file
    args.features_path = "{}/classification/features/max_{}/".format(
        args.project_file, args.max_len)

    if not os.path.exists(args.features_path):
        raise ValueError(
            "you must create features with classification/make_features.py "
            "prior to running this model.\n need file: {}".format(
            args.features_path))

    #get train features, and make optimizer
    if args.train:
        train_features = load_features("{}train.pkl".format(args.features_path))
        dataloader_train = make_dataloader(train_features, args.train_batch_size)
        data_loaders.append(dataloader_train)
        phases.append('train')

        num_train_steps = int(
            len(train_features) / args.train_batch_size /
            args.gradient_accumulation_steps * args.num_epochs)

        if args.num_outpout_checkpoints_train < 0:
            args.num_outpout_checkpoints_train = len(train_features) /  \
            args.train_batch_size  / (args.num_outpout_checkpoints_train * -1)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        t_total = num_train_steps

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

        args.train_len=len(train_features)
    else:
        optimizer = None

    # get val examples
    if args.val:
        val_features = load_features("{}val.pkl".format(args.features_path))
        dataloader_val = make_dataloader(val_features, args.val_batch_size)
        data_loaders.append(dataloader_val)
        phases.append('val')

        args.val_len=len(val_features)

    #outoput args
    string = ""
    for k, v in vars(args).items():
        string += "{}: {}\n".format(k, v)
    print(string)
    output = string + '\n'

    #load checkpoint
    if use_old_model:
        model, optimizer= load_checkpoint(
            "{}model".format(args.old_model_filename), model, optimizer)

        #output results from last model
        results = load_results("{}metrics.json".format(args.old_model_filename))
        string = "\n--loaded model--\ntrain loss: {}\nval loss: {}\n" \
            "num_epoch: {}\n".format(results[
            "average_train_epoch_losses"][-1], results["val_loss"][-1],
            results["epoch"])
        output += string
        print(string)

    outfile = open("{}output".format(args.output_dir), 'w')
    outfile.write(output)
    outfile.close()

    with open("{}args.json".format(args.output_dir), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

    best_model = model
    best_optimizer = optimizer

    metrics = {"accuracy": [],
               "precision": [],
               "recall": [],
               "f1": [],
               "lowest_loss": 100,
               "average_train_epoch_losses": [],
               "train_epoch_losses": [],
               "val_loss": [],
               "best_epoch": 0}

    highest_acc = 0

    for epoch in range(0, args.num_epochs):
        start = time.clock()
        string = 'Epoch: {}\n'.format(epoch)
        print(string, end='')
        output = output + '\n' + string
        metrics["epoch"] = epoch

        #if epoch == 6:
        #    model.unfreeze_embeddings()
        #    parameters = list(model.parameters())
        #    optimizer = torch.optim.Adam(
        #        parameters, amsgrad=True, weight_decay=weight_decay)

        #use when you validate before training, and what to validate on last epoch
        #if epoch == params["nb_epochs"] -1 and params["val"] and params["train"]:
        #    phases.append('val')
        #    data_loaders.append(dataloader_val)

        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
                intervals = args.num_outpout_checkpoints_train
                string = '--Train-- \n'
            else:
                model.eval()
                intervals = args.num_outpout_checkpoints_val
                string = '--Validation-- \n'

            print(string, end='')
            output = output + '\n' + string

            epoch_loss = []
            epoch_accuracy = []
            epoch_precision = []
            epoch_recall = []
            epoch_f1 = []
            j = 1

            for i, batch in enumerate(tqdm(data_loader, desc="batch")):
                batch = tuple(variable(t) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if phase == 'val':
                    outputs = model(input_ids, segment_ids, input_mask)

                epoch_loss.append(float(loss))
                average_epoch_loss = np.mean(epoch_loss)

                if phase == 'train':
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(parameters,
                    #    params["max_grad_norm"])
                    optimizer.step()
                    if (len(data_loader) / intervals)*j <= i+1:
                    #if len(data_loader) == i + 1:
                        string = (
                            'Example {:03d} | {} loss: {:.3f}'.format(
                                i, phase, average_epoch_loss))
                        #print(string, end='\n')
                        output = output + string + '\n'
                        outfile = open("{}output".format(args.output_dir), 'w')
                        outfile.write(output)
                        outfile.close()
                        j += 1
                    optimizer.zero_grad()

                else:
                    # get result metrics
                    targets = label_ids.cpu().numpy()
                    predicted = torch.argmax(outputs.view(-1, 2), -1).cpu().numpy()
                    accuracy, precision, recall, f1 = classifier_accuracy(
                        targets, predicted)
                    #print('{},{},{},{}'.format(accuracy, precision, recall,
                    # f1))
                    epoch_accuracy.append(accuracy)
                    epoch_precision.append(precision)
                    epoch_recall.append(recall)
                    epoch_f1.append(f1)
                    if (len(data_loader) / intervals) * j <= i + 1:
                        # if len(data_loader) == i + 1:
                        string = ('Example {:03d} | {} loss: {:.3f}'.format(
                                i, phase, average_epoch_loss))
                        # print(string, end='\n')
                        output = output + string + '\n'
                        average_epoch_accuracy = np.mean(epoch_accuracy)
                        average_epoch_precision = np.mean(epoch_precision)
                        average_epoch_recall = np.mean(epoch_recall)
                        average_epoch_f1 = np.mean(epoch_f1)
                        string = "Accuracy: {:.3f}\nPrecision: {:.3f}\n" \
                        "Recall: {:.3f}\nF1: {:.3f}\n".format(
                            average_epoch_accuracy, average_epoch_precision,
                            average_epoch_recall, average_epoch_f1)
                        output = output + string + '\n'
                        outfile = open("{}output".format(args.output_dir), 'w')
                        outfile.write(output)
                        outfile.close()
                        j += 1

            # print random sentence
            if phase == 'val':
                time_taken = time.clock() - start

                metrics["val_loss"].append(average_epoch_loss)
                string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                    phase, average_epoch_loss, time_taken)
                string += ' | lowest loss: {:.3f} highest accuracy:' \
                    ' {:.3f}'.format(metrics["lowest_loss"], highest_acc)
                #print(string, end='\n')
                output = output + '\n' + string + '\n'

                average_epoch_accuracy = np.mean(epoch_accuracy)
                average_epoch_precision = np.mean(epoch_precision)
                average_epoch_recall = np.mean(epoch_recall)
                average_epoch_f1 = np.mean(epoch_f1)
                metrics["accuracy"].append(average_epoch_accuracy),
                metrics["precision"].append(average_epoch_precision)
                metrics["recall"].append(average_epoch_recall)
                metrics["f1"].append(average_epoch_f1)

                if average_epoch_loss < metrics["lowest_loss"]:
                    best_model = model
                    best_optimizer = optimizer
                    metrics["best_epoch"] = epoch
                    metrics["lowest_loss"] = average_epoch_loss

                save_checkpoint("{}model".format(args.output_dir),
                                best_model, best_optimizer,
                                epoch, model, optimizer)

                with open("{}metrics.json".format(args.output_dir), 'w') as fp:
                    json.dump(metrics, fp, indent=4, sort_keys=True)

                string = "Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall:" \
                         " {:.3f}\nF1: {:.3f}\n".format(
                    average_epoch_accuracy, average_epoch_precision,
                    average_epoch_recall, average_epoch_f1)
                # print(string, end='\n')
                output = output + string + '\n'

                """
                random_idx = np.random.randint(len(dataset_val))
                sentence_1, sentence_2, labels = dataset_val[random_idx]
                batch = tuple(variable(t) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                outputs_var = model(sentence_1_var.unsqueeze(0),
                                    sentence_2_var.unsqueeze(0)) # unsqueeze
                #  to get the batch dimension
                outputs = outputs_var.squeeze(0).data.cpu().numpy()

                string = '> {}\n'.format(get_sentence_from_indices(
                    sentence_1, dataset_val.vocab, PairsDataset.EOS_TOKEN))

                string = string + u'> {}\n'.format(get_sentence_from_indices(
                    sentence_2, dataset_val.vocab, PairsDataset.EOS_TOKEN))

                string = string + u'target:{}|  P false:{:.3f}, P true:' \
                    u' {:.3f}'.format(targets, float(outputs[0]), float(outputs[1]))
                print(string, end='\n\n')
                output = output + string + '\n' + '\n'
                """
            else:
                metrics["average_train_epoch_losses"].append(average_epoch_loss)
                metrics["train_epoch_losses"].append(epoch_loss)

            outfile = open("{}output".format(args.output_dir), 'w')
            outfile.write(output)
            outfile.close()


if __name__ == '__main__':
    main()