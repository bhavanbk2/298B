import os
import re
import string
import random
import PyPDF2
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Ensure all necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# # Initialize Transformers models
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2-medium')
# model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')

def extract_text_from_pdf(pdf_file_path):
    extracted_text = ""
    try:
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
    except Exception as e:
        print(f"Error occurred while processing PDF '{pdf_file_path}': {e}")
    return extracted_text

def clean_text(text):
    text = text.lower()
    text = replace_contractions(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s\d]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def replace_contractions(text):
    contractions_dict = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group(0)], text)

def advanced_text_processing(text):
    synonyms_text = synonym_replacement(text)
    back_translated_text = advanced_back_translation(synonyms_text, src_lang='en', tgt_lang='fr')
    generated_text = generate_text(back_translated_text, max_new_tokens=50)
    combined_text = text + " " + generated_text
    return combined_text

def synonym_replacement(text, n=5):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').replace('-', ' ').lower()
            synonym = ''.join([char for char in synonym if char in string.ascii_lowercase + ' '])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Define back translation function
def advanced_back_translation(text, src_lang='en', tgt_lang='fr'):
    tokenizer_src = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    model_src = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    tokenizer_tgt = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}')
    model_tgt = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}')

    # Set pad token
    tokenizer_src.pad_token = tokenizer_src.eos_token
    tokenizer_tgt.pad_token = tokenizer_tgt.eos_token

    encoded_src = tokenizer_src(text, return_tensors="pt", padding=True, truncation=True)
    translated = model_src.generate(
        input_ids=encoded_src['input_ids'],
        attention_mask=encoded_src['attention_mask'],
        pad_token_id=tokenizer_src.pad_token_id
    )
    decoded_translated = tokenizer_src.decode(translated[0], skip_special_tokens=True)

    encoded_tgt = tokenizer_tgt(decoded_translated, return_tensors="pt", padding=True, truncation=True)
    back_translated = model_tgt.generate(
        input_ids=encoded_tgt['input_ids'],
        attention_mask=encoded_tgt['attention_mask'],
        pad_token_id=tokenizer_tgt.pad_token_id
    )
    decoded_back_translated = tokenizer_tgt.decode(back_translated[0], skip_special_tokens=True)

    return decoded_back_translated

# Define text generation function
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

# Set pad token for GPT-2 tokenizer
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

def generate_text(prompt, max_new_tokens=50):
    encoded_input = tokenizer_gpt2(prompt, return_tensors='pt', padding=True, truncation=True)
    output_sequences = model_gpt2.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer_gpt2.pad_token_id,
        num_return_sequences=1
    )
    return tokenizer_gpt2.decode(output_sequences[0], skip_special_tokens=True)
