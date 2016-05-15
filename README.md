# Twipsy : Heirarchical classification pipeline to monitor alcohol consumption on Twitter.

![this is a thing](http://drinkwiththewench.com/wp-content/uploads/2011/12/Twitter-beer_small.gif)


## Error Analysis

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from pipelines.helpers import ItemGetter

    clf = Pipeline([
        ("getter", ItemGetter("text")),
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())])

    clf_params = {
        'clf__C': 200,
        'clf__dual': False,
        'clf__max_iter': 100,
        'clf__multi_class': 'ovr',
        'clf__penalty': 'l2',
        'tfidf__tokenizer':tokenizer.tokenize,
        'tfidf__ngram_range':(1, 3),
        'tfidf__max_features':200000
    }

    clf.set_params(**clf_params)


### Alchohol

    AUC: .88
    F1 : .86

    52.448404   drunk
    42.239226   beer
    36.210345   drinking
    28.322088   wine
    25.026860   drink
    19.601554   alcohol
    19.076516   vodka
    18.575889   liquor
    -9.357134   starbucks
    -9.644866   sale
    -9.719909   drunk no
    -10.440220  soda
    -11.650809  tea
    -13.597104  water
    -13.606395  coffee 

    confusion matrix:

### First Person Alcohol

    AUC: .76
    F1 : .78

    coef
    16.550536   i
    15.910636   drinking
    14.836835   drinking a
    12.740684   i'm
    10.045915   drunk
    9.309862    after
    9.085708    with my
    8.248262    tonight
    8.196422    by
    -7.297446   alcohol
    -7.861017   her
    -7.913132   your
    -9.451158   they
    -9.882489   people
    -10.045302  he
    -10.088951  she
    -10.710835  you

### First Person Alcohol Level

            present     future      past
    AUC:        .78        .81       .78
    F1 :      

    Confusion matrix:

    [[763, 160, 151],
     [136, 370,  62],
     [159,  59, 238]]

#### Present

    drinking a        12.128972
    drunk             10.553835
    last drink         7.592543
    got drunk last     6.272809
    drink drink        5.853540
    i'm drunk          5.694900
    cocktail           5.673984
    drank a lot        5.559657
    so drunk           5.492124
    drinking           5.194043
    dinner             5.079105
    can drink to       4.949301
    tweet              4.937588
    today             -5.633235
    start drinking    -5.651665
    to get drunk      -5.757501
    i wanna           -5.823834
    weekend           -5.924328
    just want         -6.191953
    last night        -6.904581
    again             -6.911105
    for               -6.994582
    want              -7.158710
    get drunk         -7.757137
    need              -8.902372
    when              -8.943809
    who               -8.968795
    wanna             -9.009871
    was               -9.256000
    get drunk .      -10.094595
    tonight          -10.577908

## Future 

    tonight           18.096313
    wanna             14.631638
    need              14.420481
    get drunk         11.869691
    want               9.984220
    gonna              9.431688
    today              8.974590
    get drunk .        8.795273
    i need             8.742456
    week               8.373515
    some               8.373406
    tomorrow           8.220935
    i wanna            7.654625
    start drinking     7.491097
    wish               7.211005
    come               7.127874
    drinks             7.018599
    for                7.015012
    a drink            6.939805
    let                6.925948
    beer               6.801807
    i want             6.746357
    to get drunk       6.344923
    beer me            6.229578
    feeling           -4.912179
    drank a           -4.918264
    making            -4.953374
    today at          -4.962927
    beer is           -5.138651
    i'm drunk         -5.667315
    drink a beer      -5.899668
    was               -6.455737
    drank             -6.559461
    i drink           -8.126595
    drinking         -11.451229
    drunk            -17.432698


#### Past 

    was                  14.985710
    when                 11.988900
    last                  9.919666
    again                 9.268660
    last night            8.635193
    drinking and          7.608206
    had                   7.514924
    alcohol               7.007625
    yesterday             6.521351
    i've                  6.184506
    had a                 6.179420
    never drinking        6.032799
    after                 5.942911
    stop drinking lol     5.786537
    wish i               -4.171253
    i want               -4.194806
    i need               -4.222338
    wish                 -4.327546
    i feel drunk         -4.412836
    a drink              -4.476768
    last drink           -4.618638
    tomorrow             -4.711800
    drinks               -5.118043
    wanna                -6.408727
    drinking a           -6.622725
    got drunk last       -7.585177
    tonight              -8.148855
