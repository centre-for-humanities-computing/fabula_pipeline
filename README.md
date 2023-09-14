# fabula_pipeline

## tl;dr
This pipeline (currently) takes a folder containing books in .txt format and outputs a JSON-file containing the different literary features developed / used by the Fabula-NET team at Aarhus University.

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

 
