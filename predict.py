import os
import pandas as pd
import sys
import torch
from transformers import AutoTokenizer

from config import config
from Trainer import LightningModel
from dataset.dataset import DADataset, act_label_names
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn import metrics

class DialogClassifier:
    """
    Class to perform inference from a pre-saved checkpoint
    """

    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = LightningModel(config=config)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['state_dict'])

    #def get_classes(self):
        #return self.model.classes

    def dataloader(self, data):
        act_tag = ["" for i in range(len(data))]
        d = {'DamslActTag': act_tag, 'Text': data}
        test_data = pd.DataFrame(data=d)
        test_dataset = DADataset(tokenizer=self.tokenizer, data=test_data, max_len=self.config['max_len'],
                                 text_field=self.config['text_field'], label_field=self.config['label_field'], device=self.device)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False,
                                 num_workers=0)  # config['num_workers'])
        batch = next(iter(test_loader))
        batch['label'] = torch.Tensor([0])
        return batch


        if not isinstance(data, list):
            data = list(data)

        inputs = dict()

        input_encoding = self.tokenizer.batch_encode_plus(
            data,
            truncation=True,
            max_length=self.config['max_len'],
            return_tensors='pt',
            return_attention_mask=True,
            padding='max_length',
        ).to(self.device)
        seq_len = [len(self.tokenizer.tokenize(utt)) for utt in data]
        inputs['input_ids'] = input_encoding['input_ids'].squeeze()
        inputs['attention_mask'] = input_encoding['attention_mask'].squeeze()
        inputs['seq_len'] = torch.Tensor(seq_len)

        return inputs

    def predict(self, df):
        input = self.dataloader(df)
        with torch.no_grad():
            # model prediction labels
            outputs = self.model.model(input).argmax(dim=-1).tolist()
        return outputs

    def get_classes(self):
        test_data = pd.read_csv(
            os.path.join(config['data_dir'], config['dataset'], config['dataset'] + "_train.csv"))
        test_dataset = DADataset(tokenizer=self.model.tokenizer, data=test_data, max_len=config['max_len'],
                                 text_field=self.model.config['text_field'], label_field=config['label_field'], device=self.device)
        classes = test_dataset.label_dict()
        inv_classes = {v: k for k, v in classes.items()}  # Invert classes dictionary
        return inv_classes

    def test_dataloader(self):
        test_data = pd.read_csv(os.path.join(self.config['data_dir'], self.config['dataset'], self.config['dataset']+"_test.csv"))
        test_dataset = DADataset(tokenizer=self.tokenizer, data=test_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'], device=self.device)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)#self.config['num_workers'])
        return test_loader

    def eval(self):
        loader = self.test_dataloader()
        self.model.eval()
        outputs = []
        targets = []
        for batch in tqdm(loader):
            input_ids, attention_mask, target = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
            logits = self.model(batch).argmax(dim=-1).tolist()
            outputs.extend(logits)
            targets.extend(target)
        metrics.accuracy_score(targets, outputs)
        print(metrics.classification_report(targets, outputs))



def main(argv):
    """
    Predict speech acts for the utterances in input file
    :param argv: Takes 1 argument. File with utterances to classify, one per line.
    :return: Prints file with utterances tagged with speech act
    """

    input_file = argv[0]
    ckpt_path = 'checkpoints/epoch=28-val_accuracy=0.746056.ckpt'  # Modify to use your checkpoint
    ckpt_path = '//mnt/d/Programs/NLP/utils/CASA-Dialogue-Act-Classifier-main/output/epoch=29-val_accuracy=0.751411.ckpt'

    clf = DialogClassifier(checkpoint_path=ckpt_path, config=config)  # Choose 'cuda' if desired
    inv_classes = clf.get_classes()

    with open(input_file, 'r') as fi:
        utterances = fi.read().splitlines()
    utterances = ["hi, how are you?"]#,"I'm ok"]

    predictions = clf.predict(utterances)
    predicted_acts = [inv_classes[prediction] for prediction in predictions]

    results = pd.DataFrame(list(zip(predicted_acts, utterances)), columns=["DamslActTag", "Text"])
    filename = os.path.basename(input_file)
    results.to_csv(os.path.splitext(filename)[0] + ".out", index=False)

    print("-------------------------------------")
    print("Predicted Speech Act, Utterance")
    print("-------------------------------------")

    for utterance, prediction in zip(utterances, predicted_acts):
        for index, act_tag in enumerate(act_label_names['act_tag']):
            if act_tag == prediction:
                print(f"{prediction}({utterance})-> {act_label_names['name'][index]}")

    print("-------------------------------------")
    print("Eval on Test dataset")
    print("-------------------------------------")
    clf.eval()

if __name__ == '__main__':
    main(sys.argv[1:])
