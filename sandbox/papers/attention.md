# Attention Is All You Need

## Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional
neural networks that include an encoder and a decoder. The best performing models also
connect the encoder and decoder through an attention mechanism. We propose a new simple
network architecture, the Transformer, based solely on attention mechanisms, dispensing
with recurrence and convolutions entirely.

## Self-Attention
Self-attention, sometimes called intra-attention, is an attention mechanism relating
different positions of a single sequence in order to compute a representation of the
sequence. Self-attention has been used successfully in a variety of tasks including reading
comprehension, abstractive summarization, textual entailment and learning task-independent
sentence representations.

## Multi-Head Attention
Instead of performing a single attention function with d_model-dimensional keys, values
and queries, we found it beneficial to linearly project the queries, keys and values h times
with different, learned linear projections to dk, dk and dv dimensions respectively. On
each of these projected versions of queries, keys and values we then perform the attention
function in parallel, yielding dv-dimensional output values.

## Scaled Dot-Product Attention
We call our particular attention Scaled Dot-Product Attention. The input consists of queries
and keys of dimension dk, and values of dimension dv. We compute the dot products of the
query with all keys, divide each by sqrt(dk), and apply a softmax function to obtain the
weights on the values. The scaling factor of 1/sqrt(dk) counteracts the effect of the dot
products growing large in magnitude.

## Positional Encoding
Since our model contains no recurrence and no convolution, in order for the model to make
use of the order of the sequence, we must inject some information about the relative or
absolute position of the tokens in the sequence. To this end, we add positional encodings
to the input embeddings at the bottoms of the encoder and decoder stacks.

## Results
On the WMT 2014 English-to-German translation task, the big transformer model outperforms
the best previously reported models including ensembles by more than 2.0 BLEU, establishing
a new state-of-the-art BLEU score of 28.4. The big transformer model achieves 41.0 BLEU on
the WMT 2014 English-to-French translation task.
