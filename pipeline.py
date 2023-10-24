"""
 this pipeline assumes that books are saved as txt-files within a folder :)
 
TO DO:
[ ] setup try/except for roget
[X] implement language argument
[X] implement danish pipeline 
    needs to be tested tho (6-10-2023)
[ ] fix pandas SettingWithCopyWarning
[ ] the roget categories don't seem right or ?? 
[ ] make utils script? 

"""
import argparse
import bz2
from collections import Counter
import gzip
import json
from math import log
import re
from pathlib import Path

from afinn import Afinn
from lexical_diversity import lex_div as ld
import neurokit2 as nk
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import spacy
import textstat

import saffine.multi_detrending as md
import roget.roget as roget


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="FABULA-NET pipeline")
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="output/")
    parser.add_argument("--language", "-lang", type=str, default="english")

    return parser


def extract_text(filename: str) -> str:
    """
    read the text from given filename
    """
    with open(filename, "r") as f:
        text = f.read()
        return text


def avg_wordlen(words: list[str]) -> float:
    """
    calculates average wordlength from a list of words
    """
    len_all_words = [len(word) for word in words]
    avg_word_length = sum(len_all_words) / len(words)
    return avg_word_length


def avg_sentlen(sents: list[str]) -> float:
    """
    calculates average sentence length from a list of sentences
    """
    avg_sentlen = sum([len(sent) for sent in sents]) / len(sents)
    return avg_sentlen


def compressrat(sents: list[str]):
    """
    Calculates the GZIP compress ratio and BZIP compress ratio for the first 1500 sentences in a list of sentences
    """
    # skipping the first that are often title etc
    selection = sents[2:1502]
    asstring = " ".join(selection)  # making it a long string
    encoded = asstring.encode()  # encoding for the compression

    # GZIP
    g_compr = gzip.compress(encoded, compresslevel=9)
    gzipr = len(encoded) / len(g_compr)

    # BZIP
    b_compr = bz2.compress(encoded, compresslevel=9)
    bzipr = len(encoded) / len(b_compr)

    return gzipr, bzipr


def get_sentarc(sents: list[str]) -> list[float]:
    """
    Create a sentiment arc from a list of sentences
    """
    # code taken mainly from figs.py
    sid = SentimentIntensityAnalyzer()

    arc = []
    for sentence in sents:
        compound_pol = sid.polarity_scores(sentence)["compound"]
        arc.append(compound_pol)

    return arc


def integrate(x: list[float]) -> np.matrix:
    return np.mat(np.cumsum(x) - np.mean(x))


def divide_segments(arc: list[float], n: int):
    """
    divide a list of floats into segments of the specified number of items
    """
    for i in range(0, len(arc), n):
        yield arc[i : i + n]


def get_segment_sentmeans(arc: list[float]) -> list[float]:
    """
    get the mean sentiment for each of the 20 segments of a sentiment arc (list of floats).
    """
    n_seg_items = len(arc) // 20
    segments = divide_segments(arc, n_seg_items)

    segment_means = [np.mean(segment) for segment in segments]
    return segment_means


def get_basic_sentarc_features(arc: list[float]):
    """
    calculates basic features of the sentiment arc.
    """
    # basic features
    mean_sent = np.mean(arc)
    std_sent = np.std(arc)

    # split into 20 segments and get mean for each segment
    segment_means = get_segment_sentmeans(arc)

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


def get_hurst(arc: list[float]):
    y = integrate(arc)
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


def cleaner(text: str, lower=False) -> str:
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r'[,.;:"?!*()\']', "", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)

    if lower:
        text = text.lower()
    return text


def text_entropy(text: str, language: str, base=2, asprob=True, clean=True):
    if clean:
        text = cleaner(text)

    words = word_tokenize(text, language=language)
    total_len = len(words) - 1
    bigram_transform_prob = Counter()
    word_transform_prob = Counter()

    # Loop through each word in the cleaned text and calculate the probability of each bigram
    for i, word in enumerate(words):
        if i == 0:
            word_transform_prob[word]+=1

            # very first word gets assigned as first pre
            pre = word
            continue

        word_transform_prob[word] += 1
        bigram_transform_prob[(pre, word)] += 1
        pre = word

    # return transformation probability if asprob is set to true
    if asprob:
        return transform_prob
    # if not, calculate the entropy and return that
    if not asprob:
        log_n = log(total_len, base)

        bigram_entropy = cal_entropy(base, log_n, bigram_transform_prob)
        word_entropy = cal_entropy(base, log_n, word_transform_prob)

        return bigram_entropy / total_len, word_entropy / total_len


def cal_entropy(base, log_n, transform_prob):
    entropy = sum([-x * (log(x, base) - log_n) for x in transform_prob.values()])
    return entropy


def text_readability(text: str):
    flesch_grade = textstat.flesch_kincaid_grade(text)
    flesch_ease = textstat.flesch_reading_ease(text)
    smog = textstat.smog_index(text)
    ari = textstat.automated_readability_index(text)
    dale_chall_new = textstat.dale_chall_readability_score_v2(text)

    return flesch_grade, flesch_ease, smog, ari, dale_chall_new


def get_spacy_attributes(token):
    # Save all token attributes in a list
    token_attributes = [
        token.i,
        token.text,
        token.lemma_,
        token.is_punct,
        token.is_stop,
        token.morph,
        token.pos_,
        token.tag_,
        token.dep_,
        token.head,
        token.head.i,
        token.ent_type_,
    ]

    return token_attributes


def create_spacy_df(doc_attributes: list) -> pd.DataFrame:
    df_attributes = pd.DataFrame(
        doc_attributes,
        columns=[
            "token_i",
            "token_text",
            "token_lemma_",
            "token_is_punct",
            "token_is_stop",
            "token_morph",
            "token_pos_",
            "token_tag_",
            "token_dep_",
            "token_head",
            "token_head_i",
            "token_ent_type_",
        ],
    )
    return df_attributes


def filter_spacy_df(df: pd.DataFrame) -> pd.DataFrame:
    spacy_pos = ["NOUN", "VERB", "ADJ", "INTJ"]

    filtered_df = df.loc[
        (df["token_is_punct"] == False)
        & (df["token_is_stop"] == False)
        & (df["token_pos_"].isin(spacy_pos))
    ]

    filtered_df["token_roget_pos_"] = filtered_df["token_pos_"].map(
        {"NOUN": "N", "VERB": "V", "ADJ": "ADJ", "INTJ": "INT"}
    )
    return filtered_df


def save_spacy_df(spacy_df, filename, out_dir) -> None:
    Path(f"{out_dir}/spacy_books/").mkdir(exist_ok=True)
    spacy_df.to_csv(f"{out_dir}/spacy_books/{filename.stem}_spacy.csv")


def get_token_categories(df: pd.DataFrame) -> str:
    token_categories = df.apply(
        lambda row: roget.categories(str(row["token_lemma_"]), row["token_roget_pos_"]),
        axis=1,
    ).to_string()

    return token_categories


def main():
    parser = create_parser()
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    nltk.download("punkt")

    if args.lang == "english":
        nltk.download("vader_lexicon")

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError(
                "en_core_web_sm not downloaded, run python3 -m spacy download en_core_web_sm"
            ) from e

    elif args.lang == "danish":
        try:
            nlp = spacy.load("da_core_news_sm")

        except OSError as e:
            raise OSError(
                "da_core_news_sm not downloaded, run python3 -m spacy download da_core_news_sm"
            ) from e

    nlp.max_length = 3500000

    master_dict = {}

    print("starting loop")
    for filename in Path(in_dir).glob("*.txt"):
        temp = {}

        text = extract_text(filename)

        sents = sent_tokenize(text, language=args.lang)
        words = word_tokenize(text, language=args.lang)

        # spacy
        spacy_attributes = []
        for token in nlp(text):
            token_attributes = get_spacy_attributes(token)
            spacy_attributes.append(token_attributes)

        spacy_df = create_spacy_df(spacy_attributes)

        save_spacy_df(spacy_df, filename, out_dir)

        # stylometrics
        # for words
        temp["word_count"] = len(words)
        temp["average_wordlen"] = avg_wordlen(words)
        temp["msttr"] = ld.msttr(words, window_length=100)

        # for sentences
        if len(sents) < 1502:
            print(f"\n{filename.name}")
            print("text not long enough for stylometrics\n")
            pass
        else:
            temp["average_sentlen"] = avg_sentlen(sents)
            (
                temp["gzipr"],
                temp["bzipr"],
            ) = compressrat(sents)

        # bigram entropy
        try:
            temp["bigram_entropy"], temp["word_entropy"] = text_entropy(
                text, language=args.lang, base=2, asprob=False
            )
        except:
            print(f"\n{filename.name}")
            print("error in bigram entropy\n")
            pass


        # doing stuff that only works in english
        if args.lang == "english":
            # basic sentiment features

            arc = get_sentarc(sents)

            if len(arc) < 60:
                print(f"\n{filename.name}")
                print("arc not long enough for basic sentiment features\n")
                pass
            else:
                (
                    temp["mean_sentiment"],
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
                print(f"\n{filename.name}")
                print("error with approximate entropy\n")
                pass

            # hurst
            try:
                temp["hurst"] = get_hurst(arc)
            except:
                print(f"\n{filename.name}")
                print("error with hurst\n")
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
                print(f"\n{filename.name}")
                print("error in readability\n")
                pass

            # roget
            all_roget_categories = roget.list_all_categories()

            roget_df = filter_spacy_df(spacy_df)

            temp["roget_n_tokens"] = len(spacy_df)
            temp["roget_n_tokens_filtered"] = len(roget_df)

            token_categories = get_token_categories(roget_df)
            doc_categories = re.findall(r"(rog\d{3} \w*)", token_categories)

            for roget_cat in all_roget_categories:
                temp[roget_cat] = doc_categories.count(roget_cat)

            temp["roget_n_cats"] = len(doc_categories)

            # save arc
            temp["arc"] = arc

        # saving it all
        master_dict[filename.stem] = temp

    print("finished loop")

    Path(out_dir).mkdir(exist_ok=True)

    print("saving file")
    with open(out_dir.joinpath("books_features.json"), "w") as f:
        json.dump(master_dict, f)
    print("done :-)")


if __name__ == "__main__":
    main()
