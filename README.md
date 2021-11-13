# Hate Speech Detection

:zap: Powered by google colab 

:bookmark_tabs: 11 Datasets collection & Combination

:heavy_check_mark: Datasets Validation & Model Evaluation

:arrow_upper_right: Use TF-IDF, GloVe, BERT based EMBEDDING

:thumbsup: ROCAUC : 0.962 & F1 : 0.896 (ELECTRA Embedding + Logistic Regression)

:zap: Inference Time : 38.16 ± 3.4 ms (ELECTRA Embedding + Logistic Regression)

:chart_with_upwards_trend: Model Intenpretation, Last hidden state visualization with PCA

:ringed_planet: BentoML


# 1. Intro

HateSpeech refers to remarks that incite hatred towards others based on a specific race, nationality, religion, gender, etc. or “hate speech” directed at the weak, which can lead to a hate crime. In particular, hate speech on SNS can spread quickly and widely because of its persistence, spread, accessibility, anonymity, and lack of regulation.

The purpose of this project is to produce a hate expression detection model that can be applied in chatting situations, and there are restrictions on the task. Because it needs to be able to react in real time, (1) it should have fast inference speed, (2) it should be able to classify it well, and (3) it should be able to perform well in general, not specific situations. In addition, (4) attempts were made to detect including the context.

In order to achieve the objective, light and fast classical machine learning techniques, BERT-based lightweight models, etc. were used, and the detection speed of the model was set to within 50ms per sentence.

# 2. Related research

### 2.1 Influence of Context

The effect of context on human judgment was analyzed through the results of behavioral experiments, and a study was conducted on whether the performance could be improved when context was added to the condition. The two main questions of this study are:

(a) Does the context influence a person's decision?

(B) Does the condition according to the situation improve the performance? Context was included by limiting it to only the previous post on the bulletin board. Not surprisingly, context influences when people make judgments about whether an utterance is hate speech or not. However, it did not lead to a significant performance improvement in detecting hate speech.

As the context played an important role in human judgment, it is clearly a necessary condition for the detection of hate speech, and it is clear that it is a topic that needs further research, but in this project, training was carried out separately.

### 2.2 Dataset compatibility

Results were different depending on which dataset was trained online. For example, when trained on the Twitter dataset, the Reddit dataset tends to be poor at detecting hate speech. This project tried to create the best combination by collecting and verifying various datasets to solve this problem and show general performance.

### 2.3 Estimated time required

In the paper proposing SqueezeBERT, which is known to effectively lighten the BERT model, the time taken for each stage of the existing BERT is summarized in a table. In this table, the time occupied by embedding is 0.26%, and the time occupied by FC (Fully Connected) stage and Softmax stage in Self Attention module is 30.2% in total. FC in Feed-forward Network Layers is 69.4% and Classifier is 0.02%. We confirmed the experimental results that feed-forward consumes a lot of time. In this project, I tried to change the feed-forward part to CNN in order to shorten the time required in the feed-forward, but it was stopped due to a time problem, and I tried to shorten the time by using a classifier as a logistic instead.

### **2.4 BiLSTM, GloVe, BERT**

**GloVe (Global Vectors for Word Representation)** is an embedding method that compensates for the shortcomings of Word2Vec developed by Google. The existing Word2Vec is a method that embeds a given word in the process of predicting the peripheral word as the central word and the central word as the peripheral word, and converts it into a vector. method. While it is easier to measure the similarity between word vectors, information about the whole word can be better reflected. There is a disadvantage in that the computational complexity is large in making a co-occurrence matrix for the learning data and calculating it, but you can use the word vector that has already been trained through the official website.

**BiLSTM (Bidirectional Long short-term memory)** uses LSTM, a model that overcomes the limitations of recurrent neural networks through the gate technique. It is a model that reflects both forward and backward results. It is advantageous for correlation analysis based on context) and is widely used in various natural language processing problems considering speed.

**BERT (Bidirectional Encoder Representations form Transformer)** is a model that actively utilizes the Encoder part of the Transformer. BERT pre-trains large data such as Wiki and Book data through Masked Language Model (MLM) and Next Sentence Prediction techniques. The great advantage of BERT is that good performance can be expected through fine tuning of the BERT model itself without the need to attach a new neural network, and the state of the art has already been achieved with fine tuning in several problems.

# 3. Datasets

The data set used is as shown in <Table 1>, and in this report, it is written as an abbreviation on the right. Many data sets in English were available. Even in the FOUNTA and Wz-Ls data sets containing only the Twitter ID, even though the existing Twitter was deleted, the total collected data reached about 400,000 when the duplicates were removed after pre-processing.

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9b2a655e-d796-46a0-98c3-11d6b970745f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T104811Z&X-Amz-Expires=86400&X-Amz-Signature=fe3c771d3a616a17a3e2dfcdc6ee1c7399c89cec761381463d61365f5266e73c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=550px/>

_\<Table 1> List of collected datasets_

**In the pre-processing process**, Twitter writes a reply leaving a RT (ReTweet), but the ID comes after the RT, and this information was removed because it was judged that it would not have much meaning. The URL address was also determined to be meaningless and removed. It was judged that emoticons were careful in revealing emotions, but had little effect on the detection of hate speech. All special characters have been removed except for ‘!’ and ‘?’. Since hash tags contain information after ‘#’, only the hash (#) has been removed. Lemmartization, stemming, and stopwords removal were tried, and the evaluation results were good, but it was decided not to include it in the final function due to time delay.

**The typo has been fixed.** By using Jamspell to modify utterances such as ‘I love youuuuuuu’ to ‘I love you’, it was expected that the utterance would have more meaning. Since the speed on the cpu is 2.6ms, if there is a performance improvement, it is enough time to tolerate. However, as a result of evaluation with TF-IDF and GloVe, it decreased in all cases. The reason is that in some cases, a word that is a major keyword of hate, such as 'cunt', was changed to cant. As a result, it is assumed that the performance is rather degraded.

Datasets are validated. It was trained and validated on one dataset with Glove BiLSTM. I thought that it would be a very good dataset if the general performance was excellent with a small dataset, so Information Density, one of the indicators used when measuring the efficiency of parameters in the computer vision field, borrowed the concept and gave a penalty according to the amount of the dataset. The method was divided by the 1/10th power of the number of data included in the dataset in the ROC AUC score. The results are visualized in <Figure 1>.

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7f040f06-273f-43cc-bf33-1e7ff744b530/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T105050Z&X-Amz-Expires=86400&X-Amz-Signature=32279c806217130e4d9db9781b4cffb5cae6383adb163773ac6fd8e28945c8c9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=400px/>


Most datasets showed general performance, but the model trained with THSD and conan+multi dataset showed less than 0.25 in all validation datasets, so it was judged that there was no general performance and was excluded from the final dataset.

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8f7405f8-d5ea-4a89-9204-c8172367d52c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T105155Z&X-Amz-Expires=86400&X-Amz-Signature=73dab12df564e237736f043681857634769ae8829ae685623813a6b8054ad138&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=550px/>

_\<Figure 1> Validation Heatmap_


**Data set combinations** were evaluated by performing 44 trials. All datasets except THSD, Conan, and Multi datasets were combined and performed once, and the number of datasets was randomly selected from 2 to 10 and performed 43 times. Each dataset was divided into a training dataset and a validation dataset, and only the training dataset was used for training and the validation dataset was used for validation. In conclusion, the data and general performance increased as the amount of data increased, and the data set that did not include THSD and Conan+Multi had the best performance and was selected as the final dataset. In <Table 2>, the evaluation results for the top 5 and bottom 5 are attached.

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/77f3bd74-0e47-47a7-82f6-61f3c667ffa7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T105548Z&X-Amz-Expires=86400&X-Amz-Signature=41e35ff9ce4bfd8e28b3cef84805d15e8509d1c2be8fbbe6f412fc966f762224&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=100%/>

_\<Table 2> Validation results_

The final data set was selected through **undersampling**. There are a total of 352,541 data sets selected above, and the data with a class of 1 (hate speech) was 77,140, which accounted for 21.88%, but the number of 1s was 0 in order to solve the class imbalance problem and to train faster due to the time limit of the project. was undersampled. Additionally, data with more than 1000 characters was deleted. A total of 149,799 data sets are finally selected. For model validation, there are 119,868 training data sets and 29,931 validation data sets.

# 4. Models

### 4**.1 Training**

This is the question that took the longest to decide. Because this decision is a matter of how to build the dataset. A method of training with parent and child sentences was considered using BERT’s special token ‘[SEP]’ token. However, considering the situation of chatting, it was judged that the posts on the bulletin board were very different in nature. We could not find a dataset for chat, so we chose the next best option.

It was decided to train in the usual way. Using the dataset selected above, in the most common way, each utterance became the independent variable X, and the model was trained with the dependent variable y for hate speech. It was based on the fact that the entire utterance becomes a hate speech even if there is a partial hate speech.

When performing an actual service, the result is output after combining 2 to 5 utterances. For example, if the previous utterance is “Hello, My name is Changwoo” and the currently written utterance is “Hey Monkey”, the sentence input to the model becomes “Hello, My name is Changwoo Hey Monkey”. If you continue to input sentences like this, it is expected that you can predict including the context.

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d4afae90-8aab-4440-8cc1-c331090abbd2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T105608Z&X-Amz-Expires=86400&X-Amz-Signature=b8bc3b482f889bdbdbbc7c4c399ddc7c20413bf2117a2d31e16dbbcafe4d72de&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=100%/>

_\<Figure 2> Visualization of the results of the last layer of each model through PCA_

### **4.2 Ablation**

TF-IDF was tried before data set combination, and when it was classified through Randomforest by removing stopwords and using stem extraction, F1 Score showed satisfactory performance of about 0.81, and there was room for improvement. However, due to the nature of hate speech where terms change frequently, TF-IDF was considered to have low scalability, so no additional attempts were made.

The model tried with the final data set can be roughly divided into (1)GloVe + [Logistic, Randomforest, LGBM], (2)BERT-based models, and (3)embedding through BERT-based model + [Logistic, Randomforest, LGBM]. .

In order to examine how the data was internally classified according to the model, the last hidden state of the trained BiLSTM+LSTM, DitilBERT, and ELECTRA models were extracted, reduced to two dimensions through PCA, and then visualized as \<Figure 2>. . Even though it is reduced to two dimensions in the last layer, it seems to have been classified well with the naked eye, and it is expected that a simple model such as a logistic will produce good enough results without attaching a time-consuming FC layer.

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8f7405f8-d5ea-4a89-9204-c8172367d52c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T105155Z&X-Amz-Expires=86400&X-Amz-Signature=73dab12df564e237736f043681857634769ae8829ae685623813a6b8054ad138&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=550px/>

_\<Table 3> Performance evaluation of the model<br>*Embedded without fine-tuning_


### **4.3 Evaluation**

The trained ELECTRA model performed best. What is surprising is that the results through the FC layer actually lowered the performance, which is presumed to be overfitting the training dataset. Logistic, the simplest model in the ELECTRA model, also showed the second best performance.

DistilBERT also showed good performance following it, and the results of embedding using this model were relatively poor because embedding was performed without finetuning. Nevertheless, it shows relatively good performance compared to embedding through GloVe, so there is a lot of room for improvement. Even after embedding with GloVe and classified as a BiLSTM model, the ROC AUC Score was 0.0937, which was much better than expected.

### **4.4 Benchmark**

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/373c0544-8adf-4099-82bc-1bf0b64506da/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T110048Z&X-Amz-Expires=86400&X-Amz-Signature=95f112754f41f1d5a115d0edc4aac052573301419a0e19353aacfd8de38d23e3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=100%/>

_\<Figure 3> Elapsed time to inference 2 or 5 sentences_

The benchmark was conducted in Colab without a hardware accelerator, and Intel(R) Xeon(R) CPU @ 2.00 GHz was assigned. The sentence used to measure the time is "Hello, My Name is Changwoo. Hello, Mind Logic. Hello, CodeStates". In order to include the context, two or five sentences were repeated a total of 5000 times, and 10% of each was removed in order to erase the abnormal measurement. Finally, 4000 pieces of measured data were evaluated. It is summarized in \<Table 4> based on the 95% confidence interval (mean ± 2 standard deviations).

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d484d16a-997d-461e-ace8-c9c26458189c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T105907Z&X-Amz-Expires=86400&X-Amz-Signature=4a332ee0affd9ef1c343796fede3f9324460adfd70f283b903d5a426400f986d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=550px/>

_\<Table 4> Elapsed time to inference 2 or 5 sentences_

Parentheses in the Model column of \<Table 4> indicate the number of sentences connected to include context. In case of ‘(2)’, prediction is made in succession of 2 sentences, and in case of ‘(5)’, prediction is continued up to 5 sentences. Compared to the FC layer classification in the existing BERT-based model, Logistic Regression, a simple model, detected the fastest. Logistic+ELECTRA, ELECTRA, and LGBM+ELECTRA that passed the standard of 50ms were a total of three, and none of the models that predicted five sentences did not arrive within the time frame. Therefore, the finally selected model was determined to be the Logistic+ELECTRA (2) model, which ranked second in performance evaluation and was the best in terms of speed.

The ELECTRA model used the Transformers API, the Logistic Regression used the Scikit-learn API, and the trained model was distributed through BentoML.

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/704fef08-67d0-48b0-85e1-19e394130ea7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211113%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211113T110002Z&X-Amz-Expires=86400&X-Amz-Signature=d5b31fd48d12165b30b98eb6cdecaa665277a12eded611ee0643c5f6c630d926&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width=100%/>

_\<Figure 4> BentoML operation screen: When class prediction is different through context_

# 5. Future research projects

### **5.1 Korean Hate Speech Detection**

Hate speech online inflicts great psychological damage on victims, which can come as a social shock. Whenever a celebrity, such as a celebrity, makes an extreme choice due to malicious comments, the phenomenon of feeling depressed throughout the community spreads and there is a possibility that an imitation attempt may occur. Recently, there was an incident where the owner of a restaurant using Coupang Eats made an extreme choice due to malicious comments and power abuse caused by fried shrimp. Due to such negative influence, many methods such as the Internet real-name system were proposed to prevent hate speech and malicious comments, but it was not enough to stop them.

Even in the YouTube video taken by a young child, it was not difficult to check the comments full of relative inferiority. Also, comments on Internet politics, entertainment, and sports articles are unpleasant to look at. The fact that you can easily access hate speech like this proves that censorship is not being done properly.

Attempts are being made to provide such censorship services. NAVER tokenizes each syllable, and provides a service called Cleanbot 2.0 through Persona embedding and ELMO transfer learning. Based on the insights gained from this project, I would like to proceed with the Korean hate expression detection project. We plan to build a data set by collecting data directly, and we will try to build a BERT-based model by composing data with the title and content of the post, the comment of the post, and the comment of the comment.

### ***Reference***

- Article19. 2015 ‘Hate Speech’ Explained A Toolkit
- National Human Rights Commission(Korea). 2016. A study on the actual condition of hate speech and regulation measures
- John Pavlopoulosy, Jeffrey Sorensenz, Lucas Dixonz, Nithum Thainz, Ion Androutsopoulosy. 2020.. Toxicity Detection: Does Context Really Matter?
- Kunze Wang, Dong Lu1, Soyeon Caren Han, Siqu Long, Josiah Poon. 2020. Detect All Abuse! Toward Universal Abusive Language Detection - Models, Kunze Wang
- Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, Kurt W. Keutzer. 2020. SqueezeBERT: What can computer vision teach NLP, about efficient neural networks?
- Jeffrey Pennington, Richard Socher, Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation
- [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- [https://github.com/bakwc/JamSpell](https://github.com/bakwc/JamSpell)
- [https://github.com/neuspell/neuspell](https://github.com/neuspell/neuspell)
