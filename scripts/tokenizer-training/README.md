# SentencePiece Tokenizer Training

This directory includes code for training SentencePiece tokenizer models on the
CPU cluster.

The main things to keep in mind when training SentencePiece models:

- while there is a multithread option, most of the work is done on a single CPU
- memory requirments are very high
- the input must be a single file

For details, see the [SentencePiece training options docs](https://github.com/google/sentencepiece/blob/master/doc/options.md). 
