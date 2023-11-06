"""
 this pipeline assumes that books are saved as txt-files within a folder :)
 
TO DO:
[ ] setup try/except for roget
[ ] fix pandas SettingWithCopyWarning
[ ] the roget categories don't seem right or ?? 

"""
import argparse
import json
from pathlib import Path

from lexical_diversity import lex_div as ld
import neurokit2 as nk
from nltk.tokenize import sent_tokenize
import nltk
import spacy

from utils import *


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="FABULA-NET pipeline")
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output/")
    parser.add_argument(
        "-lang", choices=["english", "danish"], type=str, default="english"
    )
    parser.add_argument(
        "--sentiment_method",
        "-sent",
        choices=["afinn", "vader", "syuzhet", "avg_syuzhet_vader"],
        type=str,
        default="vader",
    )
    parser.add_argument(
        "--ucloud",
        action="store_true",
        help="set ucloud to true. otherwise it assumes local",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    incompatible_args = check_args(args)

    if incompatible_args != None:
        raise ValueError(f"{incompatible_args}")

    nltk.download("punkt")

    if args.lang == "english":
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

    # checking that there are actually files in that folder
    if list(Path(in_dir).glob("*.txt")) == []:
        raise ValueError(
            "The folder specified as --in_dir containes no .txt files. Check the path is correct"
        )

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
            temp["gzipr"], temp["bzipr"] = compressrat(sents)

        # bigram and word entropy
        try:
            temp["bigram_entropy"], temp["word_entropy"] = text_entropy(
                text, language=args.lang, base=2, asprob=False
            )
        except:
            print(f"\n{filename.name}")
            print("error in bigram and/or word entropy\n")
            pass

        # setting up sentiment analyzer
        if "vader" in args.sentiment_method:
            nltk.download("vader_lexicon")

        arc = get_sentarc(sents, args.sentiment_method, args.lang)

        # basic sentiment features
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

        # doing the things that only work in English
        if args.lang == "english":
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
