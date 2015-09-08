from twokenize.twokenize import tokenize
from scipy.stats import randint

text_grid = {
    'features__text__tfidf__analyzer':
        ['char', 'word'],
    'features__text__tfidf__lowercase':
        [False, True],
    'features__text__tfidf__max_features':
        randint(70000, 100000),
    'features__text__tfidf__min_df':
        randint(1, 5),
    'features__text__tfidf__ngram_range': [
        (1, 1),
        (1, 3),
        (2, 5),
        (2, 8)],
    'features__text__tfidf__norm':
        ['l2', 'l1'],
    'features__text__tfidf__stop_words':
        [None, "english"],
    'features__text__tfidf__tokenizer':
        [None, tokenize],
}
