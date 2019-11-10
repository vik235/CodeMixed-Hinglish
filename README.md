
# CodeMixed-Hinglish
Note: The datasrt contains profanity. Any mention of examples is purely for illustration purposes. 

Contains artifacts for training few architecture of neural networks for classifying code mixed Hindi and English (Hinglish) tweets into 3 classes - Offensive, Hateful and Non offensive. 

Folder structure : 
1. Data : 
	a. Contains artifacts related to labelled dataset.
		i. HOT_Dataset_modified.csv is the source (citation will be provided later)
		ii. CuredData:
			Final massagd and transliterated data. 
		iii. labels.csv, messages.csv, messages_cleaned.csv: Different stages of data parsing until final curing. 
		iv. transmessage.csv : Final transliterated messages.
		v. labeled_data.csv: English source for the 3 categories. Later on, model will be learnt on this and then then transferred to our HOT source. 
		Source will be cited later. 
		
	b. HindiCorpus :
		
		Indic corpus that will be used later on for creating workd embeddings on the indic language, Hindi. 

2. 	Logs:
		Logs for tensorboard sent vai Kers callbacks. At the moment it just have scalars for comparing different models
		and keepign an history of training and validation set learnign/testing. 
		
3. 	Models:
		Learnt models mapped with performance shwon in tensor board. 
	
	Modelperf:
		Performance characteristics of some of the learnt models along with architecture.  

4. 	Transilteration: 
		Code in C# to use Google Translate API for translating source messages to English. Code is dirty and will be improved to correct failure mode reliability at run time. Google API had reliablity issues 
		when tried first tiem and hence batch processing with retry and logging is needed. But we have gone trhough few successful passes of the translation already so makign these changes is second priority. 
		Links: 
		API documentation: https://cloud.google.com/translate/docs/
		Setup: https://cloud.google.com/translate/docs/advanced/setup-advanced
		API reference:https://cloud.google.com/translate/docs/apis

5.  Cleansing;
	Hot;
	Preprocessing
	DataSplit;
	Model_dev_Hot; 
		These are files related to work done for model development and provides an end to end workign pipeline. During next few days/weeks these will structured and made modular 
		to support sweeping of hyper params/frictionless logging for debuggign purposes. 
		
6.	TBD:
	Clean the scripts and modularize it.
	Add hyperparam search.
	Use indic language for translitration.
	Employ LSTM (bidrectional) with/without attention
	Use emoticons
	Learn emneddings. 
	
		
		
		
		
		
	
			
		