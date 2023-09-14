# fabula_pipeline

## tl;dr
This pipeline (currently) takes a folder containing books in .txt format and outputs a JSON-file containing the different literary features developed / used by the Fabula-NET team at Aarhus University.
Currently only English is supported. 

Features outputted: 

- Stylometrics:
    - word count, mean word length, MSTTR
    - mean sentence length, GZIPR, BZIPR
- Sentiment arc:
    - mean & standard deviation
    - mean sentiment for each segment
    - mean sentiment for first 10%, last 10% 
    - difference in sentiment between last 10% and the rest of the sentiment arc
- Approximate entropy
- Hurst 
- Bigram Entripy 
- Readability:
    - flesch grade, flesch ease, smog, ari, dale chall new

## Getting Started

Install all requirements for the pipeline script.

```bash
pip install -r requirements.txt
```

The pipeline scripts assumes that the books, which should go through the pipeline, are in individual .txt-files and the filename corresponds to the book's id (5 integers) with three 0's in front. E.g.:

    book_files
        \_ 00034231.txt
        \_ 00018472.txt
        \_ 00019923.txt
        ..
        ..


## How to run the pipeline

The script has two command-line argument:
1. input directory (`--in_dir`)
2. output directory (`--out_dir`)

`--in_dir` should point to the folder containing the book files. 
`--out_dir` defaults to a folder called `output/` (which will be created if it does not exist already), but can be used to point to a different folder.

You can run the CLI like this with default output:

```bash
python3 pipeline.py --in_dir your/data/path 
```

Or you can specify a folder for the output

```bash
python3 pipeline.py --in_dir your/data/path/ --out_dir your/out/path/
```

## Output
The pipeline script will create a JSON-file in the `--out_dir` folder called `books_features.json`.

It is a nested dictionary, where the top-level key is the book id (without 0's), and the value is a dictionary where the keys are the name of the features and the values are the results for the features. 
E.g.,

{"34231": {word_count: int,
            average_wordlen: float,
            msttr: float,
            average_sentlen: float,
            gzipr: float,
            bzipr: float,
            mean_sentiment: float,
            std_sentiment: float,
            mean_sentiment_per_segment: list,
            mean_sentiment_first_ten_percent: float,
            mean_sentiment_last_ten_percent: float,
            difference_lastten_therest: float,
            approximate_entropy: list,
            hurst: float,
            bigram_entropy: float,
            flesch_grade: float,
            flesch_ease: float,
            smog: float,
            ari: float,
            dale_chall_new: float,
            arc: list}}

 
## Future implementations 

- Making an optional argument, that specifies whether goodreads features should be run as well
- Making a pipeline for Danish books 
- Adding other book features from Fabula-NET
