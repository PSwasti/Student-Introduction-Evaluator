from flask import Flask, request, render_template, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

try:
    rubric_df = pd.read_excel('rubric.xlsx')
except FileNotFoundError:
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
    feedback = {}

    scores['salutation'], feedback['salutation'] = evaluate_salutation(text)
    scores['keywords'], feedback['keywords'] = evaluate_keywords(text)
    scores['flow'], feedback['flow'] = evaluate_flow(text)
    scores['speech_rate'], feedback['speech_rate'] = evaluate_speech_rate(text, word_count, duration)
    scores['grammar'], feedback['grammar'] = evaluate_grammar(text)
    scores['vocabulary'], feedback['vocabulary'] = evaluate_vocabulary(text)
    scores['filler_words'], feedback['filler_words'] = evaluate_filler_words(text)
    scores['sentiment'], feedback['sentiment'] = evaluate_sentiment(text)

    overall_score = compute_overall_score(scores)

    return {
        'overall_score': overall_score,
        'criteria_scores': scores,
        'feedback': feedback
    }

def evaluate_salutation(text):
    text_lower = text.lower().strip()
    excellent_phrases = [
        "i am excited to introduce", "feeling great",
        "i'm excited to introduce", "feeling very great",
        "i feel great to introduce"
    ]
    for phrase in excellent_phrases:
        if phrase in text_lower:
            return 5, "Excellent energetic salutation."

    good_salutations = ["good morning", "good afternoon", "good evening", "good day", "hello everyone"]
    for sal in good_salutations:
        if sal in text_lower:
            return 4, "Good polite greeting."

    normal_salutations = ["hi", "hello"]
    for sal in normal_salutations:
        if text_lower.startswith(sal) or f"{sal} " in text_lower:
            return 2, "Basic salutation; could be stronger."

    return 0, "No clear greeting found."

def calculate_similarity(query, passages):
    query_embedding = model.encode(query)
    passage_embeddings = model.encode(passages)
    similarity = np.inner(query_embedding, passage_embeddings)
    return similarity

def evaluate_keywords(text):
    text_lower = text.lower()
    feedback = "Keywords: "
    score = 0

    essential = ["name", "age", "family", "hobbies"]
    similarities = calculate_similarity(text, essential)

    for i, sim in enumerate(similarities):
        if sim > 0.7:
            score += 4
            feedback += f"{essential[i]} (sim={sim:.2f}), "

    if "school" in text_lower or "class" in text_lower:
        score += 4
        feedback += "school/class, "

    good_passages = ["origin location", "goal", "interesting thing", "strengths"]
    good_sim = calculate_similarity(text, good_passages)

    for i, sim in enumerate(good_sim):
        if sim > 0.8:
            score += 2
            feedback += f"{good_passages[i]} (sim={sim:.2f}), "

    family_words = ['mother', 'father', 'sister', 'brother', 'parent']
    if any(word in text_lower for word in family_words):
        score += 2
        feedback += "family, "

    return score, feedback

def evaluate_flow(text):
    text_lower = text.lower()

    salutations = ["hello", "hi", "good morning", "good afternoon", "good evening"]
    basic = ['name', 'age', 'class', 'school', 'place']
    additional = ['hobbies', 'family', 'friends', 'interests', 'activities']
    closings = ['thank you', 'goodbye', 'thanks for listening', 'thank you for listening', 'that’s all']

    salutation_found = any(w in text_lower for w in salutations)
    basic_found = any(w in text_lower for w in basic)
    additional_found = any(w in text_lower for w in additional)
    closing_found = any(w in text_lower for w in closings)

    if salutation_found and basic_found and additional_found and closing_found:
        return 5, "Excellent structure and flow."
    else:
        return 0, "Flow incomplete."

def evaluate_speech_rate(text, word_count, duration):
    speech_rate = (word_count / duration) * 60
    if speech_rate > 161:
        return 2, f"Too fast ({speech_rate:.2f} WPM)."
    elif 141 <= speech_rate <= 160:
        return 6, f"Good pace ({speech_rate:.2f} WPM)."
    elif 111 <= speech_rate <= 140:
        return 10, f"Excellent pace ({speech_rate:.2f} WPM)."
    elif 81 <= speech_rate <= 110:
        return 6, f"A bit slow ({speech_rate:.2f} WPM)."
    else:
        return 2, f"Too slow ({speech_rate:.2f} WPM)."

def evaluate_grammar(text):
    matches = tool.check(text)
    errors_per_100 = (len(matches) / len(text.split())) * 100
    quality = 1 - min(errors_per_100 / 10, 1)

    if quality > 0.9:
        return 10, "Excellent grammar."
    elif quality >= 0.7:
        return 8, "Good grammar."
    elif quality >= 0.5:
        return 6, "Average grammar."
    elif quality >= 0.3:
        return 4, "Weak grammar."
    else:
        return 2, "Poor grammar."

def evaluate_vocabulary(text):
    words = text.split()
    distinct = len(set(words))
    ttr = distinct / len(words) if words else 0

    if ttr >= 0.9:
        return 10, "Excellent vocabulary."
    elif ttr >= 0.7:
        return 8, "Good vocabulary."
    elif ttr >= 0.5:
        return 6, "Average vocabulary."
    elif ttr >= 0.3:
        return 4, "Limited vocabulary."
    else:
        return 2, "Very repetitive."

def evaluate_filler_words(text):
    filler_words = [
        'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right',
        'i mean', 'well', 'kinda', 'sort of', 'okay', 'hmm', 'ah'
    ]

    words = text.lower().split()
    total = len(words)
    filler_count = sum(1 for w in words if w in filler_words)

    rate = (filler_count / total) * 100 if total > 0 else 0

    if rate <= 3:
        return 15, "Excellent filler-word control."
    elif rate <= 6:
        return 12, "Good control."
    elif rate <= 9:
        return 9, "Average."
    elif rate <= 12:
        return 6, "Too many fillers."
    else:
        return 3, "Very high filler usage."

def evaluate_sentiment(text):
    sentiment = analyzer.polarity_scores(text)['compound']

    if sentiment >= 0.9:
        return 15, f"Very positive (score {sentiment:.2f})."
    elif sentiment >= 0.7:
        return 12, f"Positive (score {sentiment:.2f})."
    elif sentiment >= 0.5:
        return 9, f"Mildly positive (score {sentiment:.2f})."
    elif sentiment >= 0.3:
        return 6, f"Neutral-positive (score {sentiment:.2f})."
    else:
        return 3, f"Neutral/negative (score {sentiment:.2f})."

def compute_overall_score(scores):
    return sum(scores.values())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)

