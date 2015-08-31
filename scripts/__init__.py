from tokenize import tokenize

text_grid = {
    'features__text__tfidf__analyzer':
        ['char', 'word'],
    'features__text__tfidf__lowercase':
        [False, True],
    'features__text__tfidf__max_features':
        list(range(70000, 120000, 5000)),
    'features__text__tfidf__min_df':
        [1, 5, 10, 20],
    'features__text__tfidf__ngram_range':[
        (1, 1),
        (1, 3),
        (2, 5),
        (2, 8)],
    'features__text__tfidf__norm':
        ['l2',  'l1'],
    'features__text__tfidf__stop_words':
        [None, "english"],
    'features__text__tfidf__strip_accents':
        [None, True],
    'fertures__text__tfidf__tokenizer':
        [None, tokenize],
}
