

from pytorch_pretrained_bert.tokenization import BertTokenizer
from classification.dataset import RedditProcessor, convert_examples_to_features
from classification.utils import save_features

def main():
    max_len = 40
    project_file =  "/home/mattd/PycharmProjects/reddit_BERT/classification"
    dataset_path = "{}/data/".format(project_file)

    processor = RedditProcessor()

    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                              do_lower_case=True)

    train_examples = processor.get_train_examples(dataset_path)

    train_features = convert_examples_to_features(
        train_examples, label_list, max_len, tokenizer)

    train_filename = "{}/features/max_{}/train.pkl".format(project_file, max_len)
    save_features(train_filename, train_features)

    val_examples = processor.get_val_examples(dataset_path)

    val_features = convert_examples_to_features(
        val_examples, label_list, max_len, tokenizer)

    val_filename = "{}/features/max_{}/val.pkl".format(project_file, max_len)
    save_features(val_filename, val_features)


if __name__ == '__main__':
    main()
