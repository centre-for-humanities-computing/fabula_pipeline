# this pipeline assumes that books are saved as txt-files within a folder :)
# A CLI that takes an input direcotry and an output directory
import argparse
import bz2
import gzip
import json
import re
from pathlib import Path

from collections import Counter
from lexical_diversity import lex_div as ld
from math import log
import neurokit2 as nk
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import textstat

import saffine.multi_detrending as md
import saffine.detrending_method as dm


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="FABULA-NET pipeline")
    parser.add_argument("in_dir", nargs="?", type=str, default="test_files/")
    parser.add_argument("out_dir", nargs="?", type=str, default="output/")

    return parser


def extract_text(filename):
    with open(filename, "r") as f:
        text = f.read()
        return text


def wordcount_wordlen_msttr(text: str):
    # tokenize
    words = word_tokenize(text)
    # Wordcount
    count = len(words)
    # Avg. wordlenght
    len_all_words = [len(word) for word in words]
    avg_word_length = sum(len_all_words) / count
    # MSTTR
    msttr = ld.msttr(words, window_length=100)  # Then calculate MSTTR-100
    return count, avg_word_length, msttr


def avgsentlen_compressrat(sents: str):
    avg_sentlen = sum([len(sent) for sent in sents]) / len(sents)

    # taking only 1500 sentences (skipping the first that are often title etc)
    selection = sents[2:1502]
    asstring = " ".join(selection)  # making it a long string
    encoded = asstring.encode()  # encoding for the compression

    # GZIP
    g_compr = gzip.compress(encoded, compresslevel=9)
    gzipr = len(encoded) / len(g_compr)

    # BZIP
    b_compr = bz2.compress(encoded, compresslevel=9)
    bzipr = len(encoded) / len(b_compr)

    return avg_sentlen, gzipr, bzipr


def get_sentarc(sents: list):
    # this is mainly form figs.py
    sid = SentimentIntensityAnalyzer()

    arc = []
    for sentence in sents:
        compound_pol = sid.polarity_scores(sentence)["compound"]
        arc.append(compound_pol)

    return arc


def integrate(x):
    return np.mat(np.cumsum(x) - np.mean(x))


def divide_segments(arc: list, n: int):
    for i in range(0, len(arc), n):
        yield arc[i : i + n]


def get_basic_sentarc_features(arc: list):
    # basic features
    mean_sent = np.mean(arc)
    std_sent = np.std(arc)

    # split into 20 segments and get mean for each segment
    n_seg_items = len(arc) // 20
    segments = divide_segments(arc, n_seg_items)

    segment_means = [np.mean(segment) for segment in segments]

    # mean of first 10%, mean of last 10%
    n_ten_items = len(arc) // 10

    mean_first_ten = np.mean(arc[:n_ten_items])
    mean_end_ten = np.mean(arc[-n_ten_items:])

    # difference between end 10% and the rest
    mean_rest = np.mean(arc[:-n_ten_items])
    diff_end_rest = mean_rest - mean_end_ten

    return (
        mean_sent,
        std_sent,
        segment_means,
        mean_first_ten,
        mean_end_ten,
        diff_end_rest,
    )


def get_hurst(story_arc):
    y = integrate(story_arc)
    uneven = y.shape[1] % 2
    if uneven:
        y = y[0, :-1]

    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)

    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    hurst = round(np.polyfit(x, y, 1)[0], 2)
    return hurst


def cleaner(text, lower=False):
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r'[,.;:"?!*()\']', "", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)

    if lower:
        text = text.lower()
    return text


def text_entropy(text: str, base=2, asprob=True, clean=True):
    if clean:
        text = cleaner(text)

    words = word_tokenize(text)
    total_len = len(words) - 1
    transform_prob = Counter()

    # Loop through each word in the cleaned text and calculate the probability of each bigram
    for i, word in enumerate(words):
        if i == 0:
            # very first word gets assigned as first pre
            pre = word
            continue
        transform_prob[(pre, word)] += 1
        pre = word

    # return transformation probability if asprob is set to true
    if asprob:
        return transform_prob
    # if not, calculate the entropy and return that
    if not asprob:
        log_n = log(total_len, base)
        entropy = sum([-x * (log(x, base) - log_n) for x in transform_prob.values()])
        return entropy / total_len


def text_readability(text):
    flesch_grade = textstat.flesch_kincaid_grade(text)
    flesch_ease = textstat.flesch_reading_ease(text)
    smog = textstat.smog_index(text)
    ari = textstat.automated_readability_index(text)
    dale_chall_new = textstat.dale_chall_readability_score_v2(text)

    return flesch_grade, flesch_ease, smog, ari, dale_chall_new


def main():
    print("load NLTK stuff")
    nltk.download("vader_lexicon")
    nltk.download("punkt")

    parser = create_parser()
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    master_dict = {}

    print("starting loop")
    for filename in Path(in_dir).glob("*.txt"):
        temp = {}

        text = extract_text(filename)
        sents = sent_tokenize(text)
        arc = get_sentarc(sents)

        # stylometrics
        # for words
        (
            temp["word_count"],
            temp["average_wordlen"],
            temp["msttr"],
        ) = wordcount_wordlen_msttr(text)
        # for sentences
        if len(sents) < 1502:
            print(filename.name)
            print("text not long enough")
            pass
        else:
            (
                temp["average_sentlen"],
                temp["gzipr"],
                temp["bzipr"],
            ) = avgsentlen_compressrat(sents)

        # basic sentiment features
        if len(arc) < 60:
            print(filename.name)
            print("arc not long enough")
            pass
        else:
            (
                temp["mean sentiment"],
                temp["std_sentiment"],
                temp["mean_sentiment_per_segment"],
                temp["mean_sentiment_first_ten_percent"],
                temp["mean_sentiment_last_ten_percent"],
                temp["difference_lastten_therest"],
            ) = get_basic_sentarc_features(arc)

        # approximate entropy
        try:
            temp["approximate_entropy"] = nk.entropy_approximate(
                arc, dimension=2, tolerance="sd"
            )
        except:
            print(filename.name)
            print("error with approximate entropy")
            pass

        # hurst
        try:
            temp["hurst"] = get_hurst(arc)
        except:
            print(filename.name)
            print("error with hurst")
            pass

        # bigram entropy
        try:
            temp["bigram_entropy"] = text_entropy(text, 2, asprob=False)
        except:
            print(filename.name)
            print("error in bigram entropy")
            pass

        # readability
        try:
            (
                temp["flesch_grade"],
                temp["flesch_ease"],
                temp["smog"],
                temp["ari"],
                temp["dale_chall_new"],
            ) = text_readability(text)

        except:
            print(filename.name)
            print("error in readability")
            pass

        # save arc
        temp["arc"] = arc

        # saving it all
        master_dict[filename.stem[3:]] = temp
        print("-")

    print("finished loop")

    Path(out_dir).mkdir(exist_ok=True)

    print("saving file")
    with open(out_dir.joinpath("books_features.json"), "w") as f:
        json.dump(master_dict, f)
    print("done :-)")


if __name__ == "__main__":
    main()
