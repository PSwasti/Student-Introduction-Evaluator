from flask import Flask, request, render_template, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

try:
    rubric_df = pd.read_excel('rubric.xlsx')
except FileNotFoundError:
    print("Error: rubric.xlsx file not found.")
    rubric_df = pd.DataFrame()

tool = language_tool_python.LanguageTool('en-US')
analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score_transcript():
    transcript = request.form['transcript']
    word_count = len(transcript.split())
    duration = 52
    results = evaluate_transcript(transcript, word_count, duration)
    return jsonify(results)

def evaluate_transcript(text, word_count, duration):
    scores = {}
    scores['salutation'] = evaluate_salutation(text)
    scores['keywords'] = evaluate_keywords(text)
    scores['flow'] = evaluate_flow(text)
    scores['speech_rate'] = evaluate_speech_rate(text, word_count, duration)
    scores['grammar'] = evaluate_grammar(text)
    scores['vocabulary'] = evaluate_vocabulary(text)
    scores['filler_words'] = evaluate_filler_words(text)
    scores['sentiment'] = evaluate_sentiment(text)
    overall_score = compute_overall_score(scores)
    return {'overall_score': overall_score, 'criteria_scores': scores}

def evaluate_salutation(text):
    text_lower = text.lower().strip()
    excellent_phrases = ["i am excited to introduce", "feeling great", "i'm excited to introduce", "feeling very great", "i feel great to introduce"]
    for phrase in excellent_phrases:
        if phrase in text_lower:
            return 5
    good_salutations = ["good morning", "good afternoon", "good evening", "good day", "hello everyone"]
    for sal in good_salutations:
        if sal in text_lower:
            return 4
    normal_salutations = ["hi", "hello"]
    for sal in normal_salutations:
        if text_lower.startswith(sal) or f"{sal} " in text_lower:
            return 2
    return 0

def evaluate_keywords(text):
    essential_keywords = ['name', 'age', 'family', 'hobbies', 'goals']
    score = 0
    for keyword in essential_keywords:
        if keyword in text.lower():
            score += 4
    if 'school' in text.lower() or 'class' in text.lower():
        score += 4
    good_keywords = ['interest', 'passion', 'experience', 'skills', 'background']
    for keyword in good_keywords:
        if keyword in text.lower():
            score += 2
    if any(family_member in text.lower() for family_member in ['mother', 'father', 'sister', 'brother', 'parent']):
            score += 2
    return score

def evaluate_flow(text):
    text_lower = text.lower()
    salutations = ["hello", "hi", "good morning", "good afternoon", "good evening", "greetings", "hey"]
    basic_details_keywords = ['name', 'age', 'class', 'school', 'place']
    additional_details_keywords = ['hobbies', 'family', 'friends', 'interests', 'activities']
    closing_keywords = ['thank you', 'goodbye', 'thanks for listening', 'thank you for listening', 'that’s all']
    salutation_found = False
    basic_details_found = False
    additional_details_found = False
    closing_found = False
    for salutation in salutations:
        if salutation in text_lower:
            salutation_found = True
            break
    if salutation_found:
        basic_details_found = any(keyword in text_lower for keyword in basic_details_keywords)
    if basic_details_found:
        additional_details_found = any(keyword in text_lower for keyword in additional_details_keywords)
    if additional_details_found:
        closing_found = any(keyword in text_lower for keyword in closing_keywords)
    if salutation_found and basic_details_found and additional_details_found and closing_found:
        return 5
    else:
        return 0

def evaluate_speech_rate(text, word_count, duration):
    speech_rate = (word_count / duration) * 60
    if speech_rate > 161:
        return 2
    elif 141 <= speech_rate <= 160:
        return 6
    elif 111 <= speech_rate <= 140:
        return 10
    elif 81 <= speech_rate <= 110:
        return 6
    else:
        return 2

def evaluate_grammar(text):
    matches = tool.check(text)
    errors_per_100_words = len(matches) / len(text.split()) * 100
    grammar_score = 1 - min(errors_per_100_words / 10, 1)
    if grammar_score > 0.9:
        return 10
    elif grammar_score >= 0.7:
        return 8
    elif grammar_score >= 0.5:
        return 6
    elif grammar_score >= 0.3:
        return 4
    else:
        return 2

def evaluate_vocabulary(text):
    words = text.split()
    distinct_words = len(set(words))
    ttr = distinct_words / len(words) if len(words) > 0 else 0
    if ttr >= 0.9:
        return 10
    elif ttr >= 0.7:
        return 8
    elif ttr >= 0.5:
        return 6
    elif ttr >= 0.3:
        return 4
    else:
        return 2

def evaluate_filler_words(text):
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'i mean', 'well', 'kinda', 'sort of', 'okay', 'hmm', 'ah']
    word_count = len(text.split())
    filler_word_count = sum(1 for word in text.split() if word in filler_words)
    filler_rate = (filler_word_count / word_count) * 100 if word_count > 0 else 0
    if filler_rate <= 3:
        return 15
    elif filler_rate <= 6:
        return 12
    elif filler_rate <= 9:
        return 9
    elif filler_rate <= 12:
        return 6
    else:
        return 3

def evaluate_sentiment(text):
    sentiment = analyzer.polarity_scores(text)['compound']
    if sentiment >= 0.9:
        return 15
    elif sentiment >= 0.7:
        return 12
    elif sentiment >= 0.5:
        return 9
    elif sentiment >= 0.3:
        return 6
    else:
        return 3

def compute_overall_score(scores):
    total_score = sum(scores.values())
    return total_score

if __name__ == '__main__':
    app.run(debug=True)
