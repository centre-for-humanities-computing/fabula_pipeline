# fabula_pipeline

## tl;dr
This pipeline takes a folder containing books in .txt format and outputs a JSON-file containing the different literary features developed / used by the Fabula-NET team at Aarhus University.
Currently only English and Danish are supported. 

Features given no matter the language: 

- Stylometrics:
    - word count, mean word length, MSTTR
    - mean sentence length, GZIPR, BZIPR
- Bigram entropy and word entropy
- Sentiment arc:
    - mean & standard deviation
    - mean sentiment for each segment (arc divided into 20 segments)
    - mean sentiment for first 10%, mean sentiment for last 10% 
    - difference in sentiment between last 10% and the rest of the sentiment arc
    - Approximate entropy
    - Hurst

Features given only for English: 
- Readability:
    - flesch grade, flesch ease, smog, ari, dale chall new
- Roget Categories


Additionally, for each text a CSV-file is produced that contains the token attributes from SpaCy which is saved in a folder called 'spacy_books/' within the specified output folder.


## Getting Started

Install all requirements for the pipeline script.

```bash
pip install -r requirements.txt
```

The pipeline scripts assumes that the books, which should go through the pipeline, are in individual .txt-files with unique filenames, e.g.,: 

    book_files
        \_ 00000001.txt
        \_ 00000002.txt
        \_ 00000003.txt
        ..
        ..


## How to run the pipeline

The pipeline has five command-line argument:
1. input directory (`--in_dir`)
2. output directory (`--out_dir`)
3. language (`-lang`)
4. sentiment method (`--sentiment_method` or `sent`)
5. ucloud (`--ucloud`)


`--in_dir` should point to the folder containing the book files. 

`--out_dir` specifies the folder the book features and SpaCy attributes are saved to. It defaults to a folder called `output/` (which will be created if it does not exist already), but can be used to point to a different folder.

`-lang` specifies which language the books are in. For now only `english` and `danish` are supported. `english` is the deafult option.

`--sentiment_method` specifies which method should be used for sentiment analysis for each sentence in the book. For now `afinn`, `vader`, `syuzhet`, and `avg_vader_syuzhet` are supported. The last options takes the mean of the two different methods for each sentence. If language is set to English, the sentiment method has to be `afinn`. 

`--ucloud` is a flag that is used if the code is run on the cloud computing service UCloud. This is because the Syuzhet sentiment analysis does not currently work on UCloud. 


To run the pipeline, go into the run_pipe.sh and set the command-line arguments to the desired values. Afterwards, the script can be run like this:

```bash
bash run_pipe.sh
```

The pipeline can of course also be run regularly in the command-line: 

```bash
python3 src/pipeline.py --in_dir your/data/path/ --out_dir your/out/path/ -lang "danish"
```

## Output
The pipeline script will create a JSON-file in the `--out_dir` folder called `books_features.json`.

It is a nested dictionary, where the top-level key is the filename wihtout the file extension, and the value is a dictionary where the keys are the name of the features and the values are the results for the features. 
E.g.,

    {"00000001": 
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

It will also create a folder called spacy_books/ within the `--out_dir` folder, where a CSV-file for each book is saved containing the SpaCy token attributes for each token in the book. 
 
## Future implementations 

- Making an optional argument, that specifies whether goodreads features should be run as well
