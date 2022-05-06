import numpy as np
import torch


def nli_evaluator(pred, personas, nli_model, nli_tokenizer, args):
    batch_nli = []
    for p in personas:
        if p != '':
            batch_nli.append( (pred, p) )
    batch_tokens = nli_tokenizer.batch_encode_plus(batch_nli, padding=True, truncation=True, max_length=500, return_tensors="pt", truncation_strategy="only_first")
    batch_tokens = batch_tokens.to(args.device)
    model_outputs = nli_model(**batch_tokens)
    batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
    batch_evids = batch_probs[:, args.nli["entailment_idx"]].tolist()
    batch_conts = batch_probs[:, args.nli["contradiction_idx"]].tolist()
    nli_batch_score = []
    for evid_score, cont_score in zip(batch_evids, batch_conts):
        nli_batch_score.append(evid_score - cont_score)
    
    return np.average(nli_batch_score)