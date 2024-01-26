from functools import partial
import time
import os
import fire
import tqdm
import json
import random
import datasets
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
import torch


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    assert os.path.exists(f'{workdir}/config.json')
    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, device):
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length + 1
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    token_ids = torch.tensor(token_ids, device=device)
    tgt_token_mask = torch.tensor(tgt_token_mask, device=device)
    
    return {
        'input_ids': token_ids[:, :-1],
        'labels': token_ids[:, 1:],
        'label_token_weights': tgt_token_mask[:, 1:]
    }


def loss_fn(batch, model):
    logits = model(input_ids=batch['input_ids']).logits

    loss = torch.nn.functional.cross_entropy(
        input=logits.reshape((-1, logits.shape[-1])),
        target=batch['labels'].reshape(-1),
        reduction='none')

    return (torch.sum(loss * batch['label_token_weights'].reshape(-1)) /
            torch.sum(batch['label_token_weights']))


def train(model, optimizer, examples, collate_fn, batch_size, desc):
    model.train()
    random.shuffle(examples)

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        loss.backward()
        optimizer.step()

        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
            loss=loss.item())


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    model.eval()
    losses = []

    for i in (prog_bar := tqdm.trange(
        0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        with torch.no_grad():
            loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(model,
             examples,
             src_key,
             tgt_key,
             tokenizer,
             model_max_length,
             device,
             desc):
    model.eval()

    gen_sents = []
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)

        while len(token_ids) <= model_max_length:
            with torch.no_grad():
                logits = model(
                    input_ids=torch.tensor([token_ids], device=device)
                ).logits[0, -1]
                gen_id = torch.argmax(logits).item()

            if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }


def main(dataset_name='bbaaaa/iwslt14-de-en-preprocess',
         model_max_length=128,
         n_epochs=20,
         batch_size=64,
         learning_rate=1e-4,
         device='cuda'):
    workdir = f'./workdir'
    os.makedirs(workdir, exist_ok=True)

    config = AutoConfig.from_pretrained('gpt2')
    ### CUSTOM CONFIG
    config.n_ctx = 128
    config.n_positions = 128
    config.n_embd = 128
    config.n_head = 4
    config.n_layer = 4
    config.vocab_size = 2000
    ###
    config.save_pretrained(workdir)
    model = GPT2LMHeadModel(config=config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    ### MAKE SMALLER
    dataset['train'] = dataset['train'][:20000]
    # dataset['validation'] = dataset['validation'][:]
    dataset['test'] = dataset['test'][:10]
    ###

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config.vocab_size,
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        device=device)
    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        validation_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            device=device,
            desc=desc)

        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        json.dump(gen_examples, open(
            f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

        eval_scores = evaluate_bleu(
            examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
        print(f'Epoch {epoch_idx}: {eval_scores}')

        json.dump(
            {'validation_loss': validation_loss, **eval_scores},
            open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))


if __name__ == '__main__':
    fire.Fire(main)