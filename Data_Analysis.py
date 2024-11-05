import data_preprocess as dp
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from textstat import textstat
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


base_dir = "C:/Project_Data"

# Process both domain and personality data using the functions defined
domain_data, medical_domain_data, finance_domain_data = dp.process_domain(base_dir)
persona_data, robert_persona_data, sanjay_persona_data = dp.process_persona(base_dir)



def analyze_sentiment(texts):
    sentiments = []
    for text in texts:
        blob = TextBlob(text)
        sentiments.append(blob.sentiment.polarity)  # Polarity score
    return pd.DataFrame(sentiments, columns=['Sentiment'])

# Sentiment analysis for the Finance domain
finance_sentiments = analyze_sentiment(finance_domain_data)
print("Finance Sentiment Analysis:", finance_sentiments.mean())

# Sentiment analysis for Robert's personality data
robert_sentiments = analyze_sentiment(robert_persona_data)
print("Robert's Sentiment Analysis:", robert_sentiments.mean())

# Sentiment analysis for Sanjay's personality data
sanjay_sentiments = analyze_sentiment(sanjay_persona_data)
print("Sanjay's Sentiment Analysis:", sanjay_sentiments.mean())

# Analyze sentiment for the Medical domain
medical_sentiments = analyze_sentiment(medical_domain_data)
print("Medical Sentiment Analysis:", medical_sentiments.mean())

def extract_keywords(texts, num_keywords=10):
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    # Get the highest scoring words
    dense = tfidf_matrix.todense()
    sorted_items = dense.argsort()[:, ::-1]
    keywords = [[feature_names[ind] for ind in row[:num_keywords]] for row in sorted_items]
    return keywords

# Extract keywords for each category
medical_keywords = extract_keywords(medical_domain_data)
finance_keywords = extract_keywords(finance_domain_data)
robert_keywords = extract_keywords(robert_persona_data)
sanjay_keywords = extract_keywords(sanjay_persona_data)

print("Medical Keywords:", medical_keywords)
print("Finance Keywords:", finance_keywords)
print("Robert's Keywords:", robert_keywords)
print("Sanjay's Keywords:", sanjay_keywords)

def perform_lda(texts, n_topics=5, n_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LDA(n_components=n_topics, random_state=0)
    lda.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics[f"Topic {topic_idx}"] = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
    return topics

# Perform LDA on each category
medical_topics = perform_lda(medical_domain_data)
finance_topics = perform_lda(finance_domain_data)
robert_topics = perform_lda(robert_persona_data)
sanjay_topics = perform_lda(sanjay_persona_data)

print("Medical Topics:", medical_topics)
print("Finance Topics:", finance_topics)
print("Robert's Topics:", robert_topics)
print("Sanjay's Topics:", sanjay_topics)

def analyze_readability(texts):
    readability_scores = []
    for text in texts:
        score = textstat.flesch_reading_ease(text)
        readability_scores.append(score)
    return pd.DataFrame(readability_scores, columns=['Readability'])

# Analyze readability for each category
medical_readability = analyze_readability(medical_domain_data)
finance_readability = analyze_readability(finance_domain_data)
robert_readability = analyze_readability(robert_persona_data)
sanjay_readability = analyze_readability(sanjay_persona_data)

print("Medical Readability Scores:", medical_readability.mean())
print("Finance Readability Scores:", finance_readability.mean())
print("Robert's Readability Scores:", robert_readability.mean())
print("Sanjay's Readability Scores:", sanjay_readability.mean())


# Ensure that NLTK's resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def clean_tokens(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens

def analyze_frequency(texts):
    # Aggregate all tokens from lists of texts
    all_tokens = []
    for text in texts:
        tokens = clean_tokens(text)
        all_tokens.extend(tokens)
    # Frequency distribution
    freq_dist = FreqDist(all_tokens)
    return freq_dist

def plot_frequency_distribution(freq_dist, title="Frequency Distribution", limit=30):
    # Plot the most common words
    freq_dist.plot(limit, title=title)
    plt.show()

# Example of Frequency Analysis on the Medical domain data
medical_freq_dist = analyze_frequency(medical_domain_data)
plot_frequency_distribution(medical_freq_dist, "Medical Domain Word Frequency")

# Repeat for Finance domain
finance_freq_dist = analyze_frequency(finance_domain_data)
plot_frequency_distribution(finance_freq_dist, "Finance Domain Word Frequency")

# Repeat for personalities
robert_freq_dist = analyze_frequency(robert_persona_data)
plot_frequency_distribution(robert_freq_dist, "Robert's Personality Word Frequency")

sanjay_freq_dist = analyze_frequency(sanjay_persona_data)
plot_frequency_distribution(sanjay_freq_dist, "Sanjay's Personality Word Frequency")

# Histogram of word counts for Medical domain
medical_word_counts = [len(text.split()) for text in medical_domain_data]
plt.figure(figsize=(8, 6))
plt.hist(medical_word_counts, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Word Count Distribution for Medical Domain')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Histogram of word counts for Finance domain
finance_word_counts = [len(text.split()) for text in finance_domain_data]
plt.figure(figsize=(8, 6))
plt.hist(finance_word_counts, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Word Count Distribution for Finance Domain')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Combine Robert and Sanjay persona word counts
persona_word_counts = {
    'Robert': [len(text.split()) for text in robert_persona_data],
    'Sanjay': [len(text.split()) for text in sanjay_persona_data]
}

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=list(persona_word_counts.values()), palette='Set2')
plt.title('Word Count Distribution for Robert vs. Sanjay Persona')
plt.xticks(ticks=[0, 1], labels=list(persona_word_counts.keys()))
plt.xlabel('Persona')
plt.ylabel('Word Count')
plt.show()

def plot_similarity_heatmap(data, labels, title):
    # Compute TF-IDF vectors for the data
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create a heatmap of document similarity
    plt.figure(figsize=(8, 8))
    sns.heatmap(similarity_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.title(f'Document Similarity Heatmap - {title}')
    plt.show()

# Plot document similarity heatmap for Medical and Finance domains
plot_similarity_heatmap(medical_domain_data, ['Doc ' + str(i+1) for i in range(len(medical_domain_data))], 'Medical Domain')
plot_similarity_heatmap(finance_domain_data, ['Doc ' + str(i+1) for i in range(len(finance_domain_data))], 'Finance Domain')

# Plot document similarity heatmap for Robert and Sanjay personas
plot_similarity_heatmap(robert_persona_data, ['Doc ' + str(i+1) for i in range(len(robert_persona_data))], 'Robert Persona')
plot_similarity_heatmap(sanjay_persona_data, ['Doc ' + str(i+1) for i in range(len(sanjay_persona_data))], 'Sanjay Persona')