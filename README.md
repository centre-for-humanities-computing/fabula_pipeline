# fabula_pipeline

## tl;dr
This pipeline takes a folder containing books in .txt format and outputs a JSON-file containing the different literary features developed / used by the Fabula-NET team at Aarhus University.
Currently only English and Danish is supported. 

Features given no matter language: 

- Stylometrics:
    - word count, mean word length, MSTTR
    - mean sentence length, GZIPR, BZIPR
- Bigram entropy 

Features given only for English:

- Sentiment arc:
    - mean & standard deviation
    - mean sentiment for each segment (arc divided into 20 segments)
    - mean sentiment for first 10%, mean sentiment for last 10% 
    - difference in sentiment between last 10% and the rest of the sentiment arc
- Approximate entropy
- Hurst 
- Readability:
    - flesch grade, flesch ease, smog, ari, dale chall new
- Roget Categorys


Additionally, for each text a CSV-file is produced that contains the token attributes from SpaCy which is saved in a folder called 'spacy_books/' within the specified output folder.


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

The script has three command-line argument:
1. input directory (`--in_dir`)
2. output directory (`--out_dir`)
3. language (`--language` or `-lang`)

`--in_dir` should point to the folder containing the book files. 
`--out_dir` defaults to a folder called `output/` (which will be created if it does not exist already), but can be used to point to a different folder.
`--language` specifies which language the books are in. For now `english` and `danish` er supported. `english` is the deafult option.

You can run the CLI like this with default output:

```bash
python3 pipeline.py --in_dir your/data/path 
```

Or you can specify a folder for the output

```bash
python3 pipeline.py --in_dir your/data/path/ --out_dir your/out/path/
```

Or you can specify a folder for the output and a language

```bash
python3 pipeline.py --in_dir your/data/path/ --out_dir your/out/path/ -lang "danish"
```

## Output
The pipeline script will create a JSON-file in the `--out_dir` folder called `books_features.json`.

It is a nested dictionary, where the top-level key is the book id (without 0's), and the value is a dictionary where the keys are the name of the features and the values are the results for the features. 
E.g.,

    {"34231": 
        {word_count: int, 
            average_wordlen: float,
            msttr: float,
            average_sentlen: float,
            gzipr: float,
            bzipr: float,
            bigram_entropy: float,
            flesch_grade: float,
            mean_sentiment: float,
            std_sentiment: float,
            mean_sentiment_per_segment: list,
            mean_sentiment_first_ten_percent: float,
            mean_sentiment_last_ten_percent: float,
            difference_lastten_therest: float,
            approximate_entropy: list,
            hurst: float,
            flesch_ease: float,
            smog: float,
            ari: float,
            dale_chall_new: float,
            arc: list
            }
        }

It will also create a folder called spacy_books/ within the `--out_dir` folder, where a CSV-file for each book is saved containing the token attributes for each token in the book. 
 
## Future implementations 

- Making an optional argument, that specifies whether goodreads features should be run as well
- Implement sentiment analysis using transformer models
