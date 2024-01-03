import torch
import nltk
import evaluate
import multiprocessing
nltk.download('punkt')
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from nltk import sent_tokenize
import math, re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, pipeline
import argparse
import pickle as pkl
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt


#### LOADING DATASETS
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cnn", help="cnn/xsum/reddit/news")
    #parser.add_argument('--seed', type=int, default=42, help="seed for experiments")
    parser.add_argument('--device', type=int, default=0, help="Choose which GPU to run.")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size.")
    args = parser.parse_args()
    return args

args = parse_args()

# Loading Datasets
print("-"*20)
print("Loading datasets....")

if args.dataset == 'cnn':      #USE
    data = load_dataset("cnn_dailymail", '3.0.0')
    article_key = 'article'
    summary_key = 'highlights'
    with open('data/cnn.pkl', 'rb') as f:
      summaries=pkl.load(f)

elif args.dataset == 'xsum':   #USE
    data = load_dataset("xsum")
    article_key = 'document'
    summary_key = 'summary'
    with open('data/xsum_capped_random.pkl', 'rb') as f:
      summaries=pkl.load(f)

elif args.dataset == 'news':   #USE
    data = load_dataset("argilla/news-summary")
    article_key = 'text'
    summary_key = 'prediction'
    data = DatasetDict({
        'train': data['test'],
        'test': data['train']})
    with open('data/news_capped_random.pkl', 'rb') as f:
      summaries=pkl.load(f)
    
elif args.dataset == 'reddit':   #USE
    data = load_dataset('reddit_tifu', 'long')
    article_key = 'documents'
    summary_key = 'tldr'
        # 80% train, 20% test + validation
    train_testvalid = data['train'].train_test_split(test_size=0.2, seed=42)
    # Split the 20% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    # gather everyone if you want to have a single DatasetDict
    data = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})
    with open('data/reddit_capped_random.pkl', 'rb') as f:
      summaries=pkl.load(f)
      
else:
    raise Exception('Invalid dataset/Dataset not found.')


#### GOLD STANDARD

data = data['test']
data


bad_index=[]
for idx,sum in enumerate(summaries):
  if not sum:
      bad_index.append(idx)

data=data.select(i for i in range(len(data)) 
                 if i not in set(bad_index))
print(data)
#data=data.select(range(100))

def generate_n_segments(a, n=10): #NEW
  k, m = divmod(len(a), n)
  return list((i*k+min(i, m),(i+1)*k+min(i+1, m)) for i in range(n))

def tokenize(example):
    example["original_article"] = example[article_key]
    example["article"] = nltk.sent_tokenize(example[article_key])
    if args.dataset == 'news':
        check=example[summary_key][0]['text']
        example["highlights"] = nltk.sent_tokenize(check)
    else:
        example["highlights"] = nltk.sent_tokenize(example[summary_key])
        
    example['segment_idxs'] = generate_n_segments(example["article"]) #NEW
    return example

data = data.map(tokenize, num_proc=multiprocessing.cpu_count())

# scorer = rouge_scorer.RougeScorer(['rouge2'],use_stemmer=True)

def get_overlap_scores(sentences, document):
    corpus = sentences + document
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(corpus)
    similarities = (tfidf * tfidf.T).toarray()

    return similarities[:len(sentences), len(sentences):]


def get_summary_indices_modified(article, summary, top_k=1, tolerance=0.1): #NEW

    scores = get_overlap_scores(summary, article)

    idx = scores.argmax(axis=1)
    false_idxs = np.where(scores.max(axis=1) == 0)
    idx = np.delete(idx, false_idxs)
    scores = np.delete(scores, false_idxs, axis=0)

    if top_k > 1 and len(article) > 1:
        search_idx = np.where((scores.max(axis=1) < 1-tolerance))
        biggest_idx = np.argpartition(scores[search_idx], -top_k)[:, -top_k:]
        unique_idx = np.concatenate((idx, biggest_idx.flatten()))
        unique_idx = np.unique(unique_idx)
    else:
        unique_idx = np.unique(idx)

    unique_idx.sort()
    return unique_idx

def summary_to_segments(example, n=10): #NEW
  article = example['article']
  summary = example['highlights']
  segment_idxs = example['segment_idxs']

  try:
    mapped_idxs = get_summary_indices_modified(article, summary)
  except Exception as e:
    # print(e)
    example['mapping']=np.random.randint(0,2,10)
    # print(np.random.randint(0,2,10))
    return example

  bin_counts = [0]*n

  for mapped_idx in mapped_idxs:
    for i,segment_idx in enumerate(segment_idxs):
      if mapped_idx >= segment_idx[0] and mapped_idx < segment_idx[1]:
        bin_counts[i] += 1
        break

  example['mapping'] = bin_counts
  return example


data = data.map(summary_to_segments, num_proc=multiprocessing.cpu_count())


def calculate_length(example):
    #Calculates lengths in number of sentences
    example["article_length"] = len(example['article'])
    example["highlights_length"] = len(example["highlights"])

    return example

data = data.map(calculate_length,num_proc=multiprocessing.cpu_count())

def get_overlap_scores(sentences, document):
    corpus = sentences + document
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(corpus)
    similarities = (tfidf * tfidf.T).toarray()
    
    return similarities[:len(sentences), len(sentences):]



article_sentences = []
for article in data['article']:
  for sentence in article:
    article_sentences.append(sentence)

highlight_sentences = []
for highlight in data['highlights']:
  for sentence in highlight:
    highlight_sentences.append(sentence)


#Pegasus    

# summ_tokenizer = AutoTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
# summ_model = AutoModelForSeq2SeqLM.from_pretrained('google/pegasus-cnn_dailymail').to('cuda')



# pipe = pipeline("summarization",model = "google/pegasus-cnn_dailymail", tokenizer = "google/pegasus-cnn_dailymail",device=args.device)


# import time
# t = time.time()
# model_outputs = pipe(data['original_article'],clean_up_tokenization_spaces=True,truncation=True,batch_size = args.batch_size)
# print(time.time()-t)


# data = data.add_column("model_summaries", model_outputs)

# def tokenize(example):
#     example["model_summaries"] = nltk.sent_tokenize(example["model_summaries"]['summary_text'].replace('<n>', ' '))
#     return example

# data = data.map(tokenize, num_proc=multiprocessing.cpu_count())

# model_outputs=[nltk.sent_tokenize(x) for x in summaries if x]
# data = data.add_column("model_summaries", model_outputs)

model_outputs=[x for x in summaries if x]
data = data.add_column("model_summaries", model_outputs)

def summary_to_segments_gen(example, n=10): #NEW
  article = example['article']
  summary = example['model_summaries']
  segment_idxs = example['segment_idxs']

  try:
    mapped_idxs = get_summary_indices_modified(article, summary)
  except Exception as e:
    # print(e)
    example['mapping_gen']=np.random.randint(0,2,10)
    # print(np.random.randint(0,2,10))
    return example

  bin_counts = [0]*n

  for mapped_idx in mapped_idxs:
    for i,segment_idx in enumerate(segment_idxs):
      if mapped_idx >= segment_idx[0] and mapped_idx < segment_idx[1]:
        bin_counts[i] += 1
        break

  example['mapping_gen'] = bin_counts
  return example


data = data.map(summary_to_segments_gen, num_proc=multiprocessing.cpu_count())


cumm_list1 = [0]*10
cumm_list2 = [0]*10

for ind, da in enumerate(data):
    y1 = da['mapping_gen']
#     print(y)
    cumm_list1 = [a+b for a,b in zip(cumm_list1, y1)]
    y2 = da['mapping']
#     print(y)
    cumm_list2 = [a+b for a,b in zip(cumm_list2, y2)]

x = [j for j in range(10)]

plt.xlabel("Segment")
plt.ylabel("#Total sentences across all summaries")
plt.title("Segment vs frequency [{}-Dollyv2]".format(args.dataset))
markers=[0,1,2,3,4,5,6,7,8,9]



if args.dataset == 'cnn':      #USE

  ax = plt.gca()
  ax.set_ylim([0, 22000])

elif args.dataset == 'xsum':   #USE

  ax = plt.gca()
  ax.set_ylim([0, 14000])

elif args.dataset == 'news':   #USE

  ax = plt.gca()
  ax.set_ylim([0, 1200])
    
elif args.dataset == 'reddit':   #USE

  ax = plt.gca()
  ax.set_ylim([0, 6000])


plt.plot(x, cumm_list1,'-rx', label="Generated", markevery=markers)
plt.plot(x, cumm_list2,'-bx', label="Gold", markevery=markers)
plt.xticks(x)
plt.legend()
plt.savefig('results/Dollyv2-{}_capped_random.png'.format(args.dataset))

highlights = []
model_s = []


for j in data['highlights']:
    highlights.append(' '.join(j))

for k in data['model_summaries']:
    model_s.append(' '.join(k))


rouge = evaluate.load('rouge')

print("==> Comparing generated summaries with gold summaries")
results = rouge.compute(predictions=model_s, references=highlights)
print(results)

def kl_divergence(p, q):
    """
    Calculates the KL divergence between two probability distributions p and q.
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

kl=kl_divergence(np.array(cumm_list1)/np.sum(cumm_list1), np.array(cumm_list2)/np.sum(cumm_list2))

results['KL_Divergence']=kl

# df=pd.DataFrame.from_dict(results)
df=pd.DataFrame([results])

df.to_csv('results/Dollyv2-{}_capped_random.csv'.format(args.dataset))

data.save_to_disk('saved_models/Dollyv2-{}_capped_random'.format(args.dataset))

