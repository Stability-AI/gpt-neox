import sys
import sentencepiece as spm

infile = sys.argv[1]
outfile = sys.argv[2]
model = sys.argv[3] # unigram or bpe

# dummy, p9995, byte fallback
spm.SentencePieceTrainer.train(input=infile, model_prefix=outfile, vocab_size=65000, train_extremely_large_corpus=True, character_coverage=0.9995, model_type=model, add_dummy_prefix=True, remove_extra_whitespaces=False, byte_fallback=True)

# dummy, char_coverage=1, fallback
#spm.SentencePieceTrainer.train(input=infile, model_prefix=outfile, vocab_size=65000, train_extremely_large_corpus=True, character_coverage=1, model_type=model, add_dummy_prefix=True, remove_extra_whitespaces=False, byte_fallback=True)

# no dummy, no fallback
#spm.SentencePieceTrainer.train(input=infile, model_prefix=outfile, vocab_size=65000, train_extremely_large_corpus=True, character_coverage=1, model_type=model, add_dummy_prefix=False, remove_extra_whitespaces=False)

# with pretokenize delimiter
#spm.SentencePieceTrainer.train(input=infile, model_prefix=outfile, vocab_size=65000, train_extremely_large_corpus=True, character_coverage=1, model_type=model, add_dummy_prefix=False, remove_extra_whitespaces=False, pretokenization_delimiter="\u200b", )

# pretokenize delimiter and sampling
#spm.SentencePieceTrainer.train(input=infile, model_prefix=outfile, vocab_size=65000, character_coverage=1, model_type=model, add_dummy_prefix=False, remove_extra_whitespaces=False, pretokenization_delimiter="\u200b", input_sentence_size=10000000, shuffle_input_sentence=True)
