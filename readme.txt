 NLP and Rule-Based Scoring and Evaluation System

This project provides a scoring and evaluation system for text transcripts using a hybrid approach: 'rule-based' methods for most criteria and an 'NLP-based' method for keyword evaluation. The system is built with Flask and supports the integration of various NLP models and libraries. The goal is to evaluate various aspects of a transcript, including greetings, flow, speech rate, grammar, vocabulary, sentiment, and the presence of essential keywords.

I)Features

1)Rule-Based Evaluation: The system uses predefined rules to evaluate specific aspects of the transcript, such as:

  * Salutation (greetings)
  * Speech flow (structure)
  * Speech rate
  * Grammar errors
  * Filler words

2)NLP-Based Evaluation: The only aspect evaluated using NLP models is:
(i)Keyword Evaluation: Identifies and scores the presence of essential and supplementary keywords using a transformer-based model (Sentence-Transformer).
(ii)Scoring Breakdown: The system provides individual scores for each evaluation criterion and an overall score based on the weighted criteria.


II)Requirements

* Python 3.x
* Flask
* pandas
* sentence-transformers (for keyword evaluation)
* language-tool-python (for grammar checking)
* vaderSentiment (for sentiment analysis)
* numpy (for similarity calculations)

III) Install Dependencies

You can install the required dependencies using `pip`:


pip install Flask pandas sentence-transformers language-tool-python vaderSentiment numpy

IV)Files

1)rubric.xlsx: (Optional) An Excel file that may contain the rubric used for rule-based evaluations such as salutation, flow, and speech rate. If not found, the system will work without this file.
2)index.html: The HTML template for rendering the user interface to submit the transcript.
3)app.py: The main Python application that runs the Flask server and handles the evaluation logic.

V)Usage

1.Ensure that 'rubric.xlsx' is in the project directory(optional). This file is used for rubric-based evaluations (salutation, flow, etc.). The system will still run if it’s not found.

2.Run the Flask server:

 python app.py

3.Access the web interface at 'http://localhost:5000/' to submit a transcript for evaluation.

4.Submit the transcript via the form to receive a detailed evaluation and scores for each criterion.

VI)Evaluation Criteria

The scoring system uses both 'rule-based' logic and 'NLP-based' evaluation. 

VII)Overall Score

The overall score is the sum of all individual criterion scores. The weight of each criterion can be adjusted depending on the application’s requirements.

VIII)Example

Input:
"Hello everyone, my name is Muskan, and I am 13 years old. I am studying in class 8 at Christ Public School. I live with my parents. My mother is very kind-hearted, and my father is very supportive. I enjoy playing basketball and reading books. Thank you for listening to my introduction!"

Output:
json
{ "overall_score": 85, "criteria_scores": {
					    "salutation": 5,
    					    "keywords": 12,
    					    "flow": 5,
    					    "speech_rate": 10,
    					    "grammar": 10,
    					    "filler_words": 15,
    					    "sentiment": 12}
								}

IX)Extending the System

This project is designed to be easily extended. You can:
1)Add more rule-based criteria (e.g., tone, coherence, structure).
2)Swap or integrate different NLP models
3)Incorporate additional NLP tasks, ike Named Entity Recognition (NER), summarization, or readability evaluation.


For further assistance, refer to the documentation of the individual libraries and models used in this project (e.g., Flask, Sentence-Transformers, LanguageTool, VADER).

