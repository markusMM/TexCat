# marmei185
DS coding challenge: TexCat

## Abstract

Main task is to create a classification endpoint for text.<br>
The initial idea is to utilize a multilingual LLM as a backend.

This approach would automatically deal with the multiple languages. <br>
However, the short length of the texts might imply other approaches (as they often consist signle or a few tokens). <br>
Nevertheless, in this approach, a compressions methods have been tested to find relevant features of the sets accpording to the categories.

## Pretrained Model

model/texcat.pkl

This model was pretrained on about 28k small texts labeled with text labels.
The text was ecoded by a HerBERT model and projected to 22 principle components with PCA.

## File Structure

scr
  text_encoding.py - for text encoding
  compression.py - for feature dimensionality reduction
  preprocessing.py - for high level pre-processings
  common.py - all common variables (infered from config.json)
  predictor.py - final prediction function
model
  texcat.pkl - categorization model
  pca_22.pkl - PCA compressoin model for 22 components
README.md - this file
app.py - the API
config.json - the default config (loading the pretrained model)
requirements.txt - the requirements running this app
.gitignore - terms to ignore by git

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
