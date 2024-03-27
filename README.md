# marmei185
DS coding challenge: TexCat

## Abstract

Main task is to create a classification endpoint for text.<br>
The initial idea is to utilize a multilingual LLM as a backend.

This approach would automatically deal with the multiple languages. <br>
However, the short length of the texts might imply other approaches (as they often consist signle or a few tokens). <br>
Nevertheless, in this approach, a compressions methods have been tested to find relevant features of the sets accpording to the categories.

## Installation

The easiest way is installing git from source:

``` bash
pip install git+https://github.com/markusMM/marmei185.git
```
Another method would be to simply clone this repository, create a new python environment and run:

``` bash
pip install -r requirements.txt
```

## running the app

The endpoint it written in Fask and can be started from deployment by running:
```bash
python app.py
```
