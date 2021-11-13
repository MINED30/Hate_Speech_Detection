## Hatespeech Detection 

### *Feature*         

:zap: Powered by google colab 

:bookmark_tabs: 11 Datasets collection & Combination

:heavy_check_mark: Datasets Validation & Model Evaluation

:arrow_upper_right: Use TF-IDF, GloVe, BERT based EMBEDDING

:thumbsup: ROCAUC : 0.962 & F1 : 0.896 (ELECTRA Embedding + Logistic Regression)

:zap: Inference Time : 38.16 ± 3.4 ms (ELECTRA Embedding + Logistic Regression)

:chart_with_upwards_trend: Model Intenpretation, Last hidden state visualization with PCA

:ringed_planet: BentoML


### *Notebook file explanation*

|Filename|Contents|
|---|---|
|HATESPEECH_01_(Build_Dataset).ipynb|Data Collection & Target transformation to binary|
|HATESPEECH_02_(Data_preparation).ipynb|Remove stopwords & Stemming|
|HATESPEECH_03_(TF_IDF).ipynb|TF-IDF, Machine learning model,|
|HATESPEECH_04_(GloVe).ipynb|glove.6B.100d, LSTM,  5-fold Cross Validation, Compare TF-IDF and GloVe|
|HATESPEECH_05_(BERT_TRIAL).ipynb|BERT based model trial|
|HATESPEECH_06_(Dataset_Validation).ipynb|Dataset combination and validation, Build final dataset|
|HATESPEECH_07_(Modeling_with_FinalDataset_01).ipynb|GloVe + BiLSTM, BERT based model training and evaluation|
|HATESPEECH_08_(Modeling_with_FinalDataset_02).ipynb|GloVe + Machine learning model, BERT embedding + Machine learning moedl training and evaluation|
|HATESPEECH_09_(Service_Prototype_&_Benchmarks).ipynb|Prototype, Benchmarks |
|HATESPEECH_10_(BentoML).ipynb|BentoML|
|HATESPEECH_11_(Model_Visualization).ipynb|Model Intenpretation, Last hidden state visualization with PCA|
|프로젝트보고서_(20210626-20210709).pdf|Report|

### *Reference*
- Article19. 2015 ‘Hate Speech’ Explained A Toolkit
- 국가인권위원회. 2016. 혐오표현 실태조사 및 규제방안 연구
- John Pavlopoulosy, Jeffrey Sorensenz, Lucas Dixonz, Nithum Thainz, Ion Androutsopoulosy. 2020.. Toxicity Detection: Does Context Really Matter? 
- Kunze Wang, Dong Lu1, Soyeon Caren Han, Siqu Long, Josiah Poon. 2020. Detect All Abuse! Toward Universal Abusive Language Detection - Models, Kunze Wang
- Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, Kurt W. Keutzer. 2020. SqueezeBERT: What can computer vision teach NLP, about efficient neural networks?
- Jeffrey Pennington, Richard Socher, Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation 
- https://nlp.stanford.edu/projects/glove/
- https://github.com/bakwc/JamSpell
- https://github.com/neuspell/neuspell

