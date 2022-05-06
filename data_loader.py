import pickle
from torch.nn.utils.rnn import pad_sequence
import os
from torch.utils.data import Dataset
import torch


def construct_conv(row, tokenizer):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row])
    conv = flatten(conv)
    return conv


class PersonaDataset(Dataset):
    def __init__(self, tokenizer, args, data_set, logger, evaluation = False, block_size=512):
        self.args = args
        self.evaluation = evaluation

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        self.prompts = [construct_conv(["What do you like?"], tokenizer), construct_conv(["What is your job?"], tokenizer)]

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = []
            
            # {c_di: {p_id: [[context, response, persona], ]}}
            for c_id in data_set:
                dialogs = []
                personalities = []
                for p_id in data_set[c_id]:
                    if p_id == 0:
                        for context, response, persona in data_set[c_id][p_id]:
                            dialogs.append([construct_conv(context, tokenizer), construct_conv([response], tokenizer)])
                    # personalities[0] is the original persona
                    persona = data_set[c_id][p_id][0][2]
                    personalities.append(construct_conv(persona, tokenizer))
                self.examples.append([dialogs, personalities])

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)   # return 1000

    def __getitem__(self, item):
        return self.examples[item]

    def collate(self, examples):
        input_ids = []
        fake_inputs = []
        last_person = examples[-1][1]
        personilaty = []
        if self.args.oracle:
            for dialog, persona in examples:
                for context, response in dialog[-4:]:
                    input_ids.append(torch.tensor(persona[0] + context + response, dtype=torch.long))
                    personilaty.append(persona[0])
                for context, response in dialog[-1:]:
                    for p in self.prompts:
                        fake_inputs.append(torch.tensor(context + p + response, dtype=torch.long))
        else: 
            if self.args.zero_shot:
                p_num = -2 if self.args.use_prompts else -4
                for dialog, persona in examples:
                    for context, response in dialog[p_num:]:
                        input_ids.append(torch.tensor(context + response, dtype=torch.long))
                        personilaty.append(persona[0])
                    for context, response in dialog[-1:]:
                        for p in self.prompts:
                            fake_inputs.append(torch.tensor(context + p + response, dtype=torch.long))
            else:
                p_num = 1 if self.args.constractive else 2
                for dialog, persona in examples:
                    for context, response in dialog[-2:]:
                        for p in persona[1: 1 + p_num]:
                            input_ids.append(torch.tensor(p + context + response, dtype=torch.long))
                            personilaty.append(persona[0])
                        for p in last_person[1: 1 + p_num]:
                            fake_inputs.append(torch.tensor(p + context + response, dtype=torch.long))
                    last_person = persona

        input_ids = pad_sequence(input_ids, batch_first=True)
        fake_inputs = pad_sequence(fake_inputs, batch_first=True)
        if not self.evaluation:
            return input_ids, fake_inputs
        else:
            return input_ids, personilaty


class DailyChatDataset(Dataset):
    def __init__(self, tokenizer, args, data_set, logger, evaluation = False, block_size=512):
        self.args = args
        self.evaluation = evaluation

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        self.prompts = [construct_conv(["What do you like?"], tokenizer), construct_conv(["What is your job?"], tokenizer)]

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = []

            for diag, persona in data_set:
                if len(diag) < 4: continue
                dialogs = [diag, diag[:-1], diag[:-2], diag[:-3]] 
                dialogs = [construct_conv(i, tokenizer) for i in dialogs]
                personas = [persona, persona[:-1], persona[:-2], persona[:-3]]
                personas = [construct_conv(i, tokenizer) for i in personas]
                self.examples.append([dialogs, personas])
            
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        if self.evaluation: 
            return len(self.examples[:250])
        return len(self.examples[:1000])   # return 1000

    def __getitem__(self, item):
        return self.examples[item]

    def collate(self, examples):
        input_ids = []
        fake_inputs = []

        if self.args.zero_shot:
            p_num = 2 if self.args.use_prompts else 4
            for dialog, persona in examples:
                for diag in dialog[: p_num]:
                    input_ids.append(torch.tensor(diag, dtype=torch.long))
                for diag in dialog[: int(p_num / 2)]:
                    for p in self.prompts:
                        fake_inputs.append(torch.tensor(diag + p, dtype=torch.long))
            input_ids = pad_sequence(input_ids[:-2], batch_first=True)
            fake_inputs = pad_sequence(fake_inputs[:-2], batch_first=True)
        else:
            p_num = 2 if self.args.constractive else 4
            for dialog, persona in examples:
                for diag in persona[: p_num]:
                    input_ids.append(torch.tensor(diag, dtype=torch.long))
                    fake_diag = list(reversed(diag))
                    fake_inputs.append(torch.tensor(fake_diag, dtype=torch.long))

            input_ids = pad_sequence(input_ids[:-4], batch_first=True)
            fake_inputs = pad_sequence(fake_inputs[:-4], batch_first=True)
        
        return input_ids, fake_inputs


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelWithLMHead
    from config import Args
    from torch.utils.data import DataLoader, SequentialSampler
    import logging

    logger = logging.getLogger(__name__)
    args = Args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

    # with open(args.input_dir, "rb") as f:
    #     train, valid, test = pickle.load(f)

    # eval_dataset = PersonaDataset(tokenizer, args, valid, logger)
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size, collate_fn=eval_dataset.collate, drop_last = True
    # )

    # for inputs, labels in eval_dataloader:
    #     print(inputs.shape)
    #     print(labels.shape)
    #     exit()


    with open('data/save/daily_chat.pickle', "rb") as f:
        dataset = pickle.load(f)

    train_set, valid_set, test_set = dataset["train"], dataset["validation"], dataset["test"]

    eval_dataset = DailyChatDataset(tokenizer, args, valid_set, logger)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size, collate_fn=eval_dataset.collate, drop_last = True
    )

    for inputs, labels in eval_dataloader:
        print(inputs.shape)
        print(labels.shape)
        exit()
