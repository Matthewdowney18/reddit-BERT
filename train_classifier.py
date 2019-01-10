import numpy as np
import torch
import torch.utils.data
import time
from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertForNextSentencePrediction
from pytorch_pretrained_bert.optimization import BertAdam

from classification.dataset import make_dataloader
from classification.utils import variable, cuda, get_sentence_from_indices, \
    save_checkpoint, load_checkpoint, \
    classifier_accuracy, load_params, load_features


def main():
    file = {
        "model_group": "/baseline",
        "model_name": "/classification_1",
        "old_model_name": None,
        "model_version": 0,
        "project_file": "/home/mattd/PycharmProjects/reddit_BERT"
            "/classification"}

    file["dataset_path"] = "{}/data/".format(file["project_file"])

    file["model_filename"] = '{}{}s{}_{}'.format(file["project_file"],
        file["model_group"], file["model_name"], file["model_version"])

    file["metrics_filename"] = '{}{}_metrics{}_{}'.format(file["project_file"],
        file["model_group"], file["model_name"], file["model_version"])

    file["output_file"] = '{}{}_outputs{}_{}'.format(file["project_file"],
        file["model_group"], file["model_name"], file["model_version"])

    # check_files(file)

    use_old_model = file["old_model_name"] is not None
    params = {}

    if use_old_model:
        file["old_model_filename"] = '{}{}s{}'.format(file["project_file"],
            file["model_group"], file["old_model_name"])
        params, old_files = load_params(file["old_model_filename"])
        use_old_model = old_files != {}

    if not use_old_model:
        params = {
            "bert_model": "bert-base-uncased",
            "batch_size": 70,
            "hidden_size": 256,
            "embedding_dim": 300,
            "pretrained_embeddings": True,
            "max_grad_norm": 5,
            "max_len": 40,
            "min_count": 2,
            "weight_decay": 0.00001,
            "learning_rate": 0.005,
            "num_labels": 2,
            "gradient_accumulation_steps":1,
            "warmup_proportion":0.1
        }

    params["train"] = True
    params["val"] = True
    params["num_training_examples"] = -1
    params["num_val_examples"] = -1
    params["nb_epochs"] = 2
    params["num_outpout_checkpoints_train"] = 100
    params["num_outpout_checkpoints_val"] = 1
    string= ""
    for k, v in file.items():
        string += "{}: {}\n".format(k, v)
    for k, v in params.items():
        string += "{}: {}\n".format(k, v)

    #print(string)
    output = string + '\n'

    model = cuda(BertForNextSentencePrediction.from_pretrained(
        params["bert_model"]))

    phases = []
    data_loaders = []

    #loads features from file that is created in make_features.py
    #it takes a long time to create features which is why there is a seperate
    #file
    file["features_path"] = "{}/features/max_{}/".format(
        file["project_file"], params["max_len"])

    # get val examples
    if params["val"]:
        val_features = load_features("{}val.pkl".format(file["features_path"]))
        dataloader_val = make_dataloader(val_features, params["batch_size"])
        data_loaders.append(dataloader_val)
        phases.append('val')

    #get train features, and make optimizer
    if params["train"]:
        train_features = load_features("{}train.pkl".format(file["features_path"]))
        dataloader_train = make_dataloader(train_features, params["batch_size"])
        data_loaders.append(dataloader_train)
        phases.append('train')

        num_train_steps = int(
            len(train_features) / params["batch_size"] /
            params["gradient_accumulation_steps"] * params["nb_epochs"])

        # prepare_model
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        t_total = num_train_steps

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=params["learning_rate"],
                             warmup=params["warmup_proportion"],
                             t_total=t_total)
    else:
        optimizer = None

    if use_old_model:
        model, optimizer= load_checkpoint(
            file["old_model_filename"], model, optimizer)

    parameters = list(model.parameters())

    lowest_loss = 100
    train_loss = []
    val_loss = []
    best_model = model
    best_optimizer = optimizer
    average_epoch_loss = 0

    metrics = {"accuracy": [],
               "precision": [],
               "recall": [],
               "f1": []}

    outfile = open(file["output_file"], 'w')
    outfile.write(output)
    outfile.close()

    highest_acc = 0

    for epoch in range(0, params["nb_epochs"]):
        start = time.clock()
        string = 'Epoch: {}\n'.format(epoch)
        print(string, end='')
        output = output + '\n' + string

        #if epoch == 6:
        #    model.unfreeze_embeddings()
        #    parameters = list(model.parameters())
        #    optimizer = torch.optim.Adam(
        #        parameters, amsgrad=True, weight_decay=weight_decay)
        if epoch == params["nb_epochs"] -1 and params["val"]:
            phases.append('val')
            data_loaders.append(dataloader_val)

        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
                intervals = params["num_outpout_checkpoints_train"]
                string = 'Train: \n'
            else:
                model.eval()
                intervals = params["num_outpout_checkpoints_val"]
                string = 'Validation \n'

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
                        outfile = open(file["output_file"], 'w')
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
                        string = (
                            'Example {:03d} | {} loss: {:.3f}'.format(
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
                        outfile = open(file["output_file"], 'w')
                        outfile.write(output)
                        outfile.close()
                        j += 1

            # print random sentence
            if phase == 'val':
                time_taken = time.clock() - start

                val_loss.append(average_epoch_loss)
                string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                    phase, average_epoch_loss, time_taken)
                string += ' | lowest loss: {:.3f} highest accuracy:' \
                    ' {:.3f}'.format(lowest_loss, highest_acc)
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

                string = "Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall:" \
                    " {:.3f}\nF1: {:.3f}\n".format(
                        average_epoch_accuracy, average_epoch_precision,
                        average_epoch_recall, average_epoch_f1)
                #print(string, end='\n')
                output = output + string + '\n'


                if average_epoch_loss < lowest_loss:
                    best_model = model
                    best_optimizer = optimizer
                    best_epoch = epoch
                    lowest_loss = average_epoch_loss

                save_checkpoint(
                    best_epoch, best_model, best_optimizer,
                    epoch, model, optimizer, train_loss, val_loss, metrics,
                    params, file)

                if average_epoch_accuracy > highest_acc:
                    highest_acc = average_epoch_accuracy
                """
                random_idx = np.random.randint(len(dataset_val))
                sentence_1, sentence_2, labels = dataset_val[random_idx]
                targets = labels
                sentence_1_var = variable(sentence_1)
                sentence_2_var = variable(sentence_2)

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
                train_loss.append(average_epoch_loss)
            outfile = open(file["output_file"], 'w')
            outfile.write(output)
            outfile.close()


if __name__ == '__main__':
    main()
