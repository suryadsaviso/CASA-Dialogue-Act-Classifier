from torch.utils.data import Dataset, DataLoader
import torch


act_label_names = {
    'name':[
        'Statement-non-opinion',
        'Acknowledge (Backchannel)',
        'Statement-opinion',
        'Agree/Accept',
        'Abandoned or Turn-Exit',
        'Appreciation',
        'Yes-No-Question',
        'Non-verbal',
        'Yes answers',
        'Conventional-closing',
        'Uninterpretable',
        'Wh-Question',
        'No answers',
        'Response Acknowledgement',
        'Hedge',
        'Declarative Yes-No-Question',
        'Other',
        'Backchannel in question form',
        'Quotation',
        'Summarize/reformulate',
        'Affirmative non-yes answers',
        'Action-directive',
        'Collaborative Completion',
        'Repeat-phrase',
        'Open-Question',
        'Rhetorical-Questions',
        'Hold before answer/agreement',
        'Reject',
        'Negative non-no answers',
        'Signal-non-understanding',
        'Other answers',
        'Conventional-opening',
        'Or-Clause',
        'Dispreferred answers',
        '3rd-party-talk',
        'Offers, Options, Commits',
        'Self-talk',
        'Downplayer',
        'Maybe/Accept-part',
        'Tag-Question',
        'Declarative Wh-Question',
        'Apology',
        'Thanking'
    ],
    'act_tag':[
        'sd',
        'b',
        'sv',
        'aa',
        '%',
        'ba',
        'qy',
        'x',
        'ny',
        'fc',
        '%',
        'qw',
        'nn',
        'bk',
        'h',
        'qy^d',
        'fo_o_fw_by_bc',
        'bh',
        '^q',
        'bf',
        'na',
        'ad',
        '^2',
        'b^m',
        'qo',
        'qh',
        '^h',
        'ar',
        'ng',
        'br',
        'no',
        'fp',
        'qrr',
        'arp_nd',
        't3',
        'oo_co_cc',
        't1',
        'bd',
        'aap_am',
        '^g',
        'qw^d',
        'fa',
        'ft'
    ],

    'example':[
        "Me, I'm in the legal department.",
        "Uh-huh.",
        "I think it's great",
        "That's exactly it.",
        "So, -",
        "I can imagine.",
        "Do you have to have any special training?",
        "[Laughter], [Throat_clearing]",
        "Yes.",
        "Well, it's been nice talking to you.",
        "But, uh, yeah",
        "Well, how old are you?",
        "No.",
        "Oh, okay.",
        "I don't know if I'm making any sense or not.",
        "So you can afford to get a house?",
        "Well give me a break, you know.",
        "Is that right?",
        "You can't be pregnant and have cats",
        "Oh, you mean you switched schools for the kids.",
        "It is.",
        "Why don't you go first",
        "Who aren't contributing.",
        "Oh, fajitas",
        "How about you?",
        "Who would steal a newspaper?",
        "I'm drawing a blank.",
        "Well, no",
        "Uh, not a whole lot.",
        "Excuse me?",
        "I don't know",
        "How are you?",
        "or is it more of a company?",
        "Well, not so much that.",
        "My goodness, Diane, get down from there.",
        "I'll have to check that out",
        "What's the word I'm looking for",
        "That's all right.",
        "Something like that",
        "Right?",
        "You are what kind of buff?",
        "I'm sorry.",
        "Hey thanks a lot"
    ]
    }

class DADataset(Dataset):
    
    __label_dict = dict()
    
    def __init__(self, tokenizer, data, text_field = "Text", label_field="DamslActTag", max_len=512, label_dict=None, device='cpu'):
        
        self.text = list(data[text_field]) #data['train'][text_field]
        self.acts = list(data[label_field]) #['train'][label_field]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

        if label_dict is None:
            # build/update the label dictionary
            classes = sorted(set(self.acts))
        
            for cls in classes:
                if cls not in DADataset.__label_dict.keys():
                    DADataset.__label_dict[cls]=len(DADataset.__label_dict.keys())
        else:
            DADataset.__label_dict = label_dict
    
    def __len__(self):
        return len(self.text)
    
    def label_dict(self):
        return DADataset.__label_dict
    
    def __getitem__(self, index):
        
        text = self.text[index]
        act = self.acts[index]
        label = DADataset.__label_dict[act]
        
        input_encoding = self.tokenizer.encode_plus(
            text=text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        ).to(self.device)
        
        seq_len = len(self.tokenizer.tokenize(text))
        
        return {
            "text":text,
            "input_ids":input_encoding['input_ids'].squeeze(),
            "attention_mask":input_encoding['attention_mask'].squeeze(),
            "seq_len":seq_len,
            "act":act,
            "label":torch.tensor([label], dtype=torch.long),
        }
