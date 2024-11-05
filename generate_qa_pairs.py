import os
from dotenv import load_dotenv
import google.generativeai as palm
import json
import re


def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def split_text_into_chunks_with_overlap(text, max_chunk_size, overlap_size):
    chunks = []
    words = text.split()
    current_chunk = ""
    for word in words:
        # Add 1 for space
        if len(current_chunk) + len(word) + 1 <= max_chunk_size:  
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            # Add overlap from previous chunk
            current_chunk = current_chunk[-overlap_size:] + word + " "  
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

#Format answers generated
def format_answer(answer):
    # Capitalize the first letter of each sentence
    sentences = re.split(r'(?<=[.!?]) +', answer)
    formatted_sentences = [sentence.capitalize() for sentence in sentences]

    # Join the sentences and add proper punctuation
    formatted_answer = ' '.join(formatted_sentences)

    return formatted_answer

def generate_qa_pairs(file_path, persona_attributes, model, max_chunk_size=4000, overlap_size=200, temperature=0.5):
    # Load text from file
    text = load_text_from_file(file_path)

    # Split text into chunks
    chunks = split_text_into_chunks_with_overlap(text, max_chunk_size, overlap_size)

    # Generate question-answer pairs
    generated_qa_pairs = []
    for persona, attributes in persona_attributes.items():
        # Initialize generated_questions as a list
        generated_questions = []
        for chunk in range(10):
            prompt = f"Generate questions based on the following passage to fine-tune LLM to respond like {attributes}:\n{chunks[chunk]}\n\n"
            try:
                response = palm.generate_text(
                    model=model,
                    prompt=prompt,
                    temperature=0,
                )
                # Append generated questions to the list
                questions = response.result.split("\n")
                # Remove numbering from questions
                questions = [question.split('. ', 1)[1] for question in questions if question]
                generated_questions.append(questions)
            except:
                continue            
        print("Generated questions......")
        # Generate answers for each generated question
        for chunk, question_chunk in zip(chunks[:10], generated_questions):
            interview = []
            for question_prompt in question_chunk:
                prompt = f"Question: {question_prompt}\nPassage: {chunk}\nPersona: {attributes}\nAnswer the question exactly like how the persna answered in the book so that the fin-tuned LLM can repsong in the same tone and slang like the persona:"
                try:
                    response = palm.generate_text(
                        model=model,
                        prompt=prompt,
                        temperature=0,
                    )
                    generated_answer = response.result.strip()
                    generated_answer = format_answer(generated_answer)
                    interview.append({"question": question_prompt, "answer": generated_answer})
                except:
                    continue
            # Combine the interview list with the respective persona's interview list
            found = False
            for qa_pair in generated_qa_pairs:
                if qa_pair["persona"] == persona:
                    qa_pair["interview"].extend(interview)
                    found = True
                    break

            # If persona not found, add a new dictionary with the persona and interview list
            if not found:
                generated_qa_pairs.append({"persona": persona, "interview": interview})
        print("Generated QA pairs......")
    return generated_qa_pairs

# Example usage:
def main():
    # Define persona attributes for both Dr. Sanjay Gupta and Robert Kiyosaki
    persona_attributes = {
        "dr_sanjay_gupta": {
            "personality": "compassionate",
            "communication_style": "informative",
            "expertise": ["neurosurgery", "medicine", "journalism"],
            "trustworthiness": "high",
            "compassion": "empathetic",
            "curiosity": "inquisitive",
            "public_advocacy": "pro-health",
            "versatility": "adaptive"
        },
        "robert_kiyosaki": {
            "personality": "insightful",
            "communication_style": "motivational",
            "expertise": ["finance", "investing", "entrepreneurship"],
            "trustworthiness": "respected",
            "vision": "financial independence",
            "innovation": "out-of-the-box",
            "philosophy": "wealth education",
            "versatility": "versatile"
        }
    }
    load_dotenv()
    # Set up API credentials
    palm.configure(api_key=os.getenv("PALM_API_KEY"))
    
    # Model name
    model = "models/text-bison-001"

    # File paths
    robert_file_path = "C:/Project_Data/cleansed_data/Robert.txt"
    sanjay_file_path = "C:/Project_Data/cleansed_data/Sanjay.txt"

    # Generate QA pairs for Robert Kiyosaki
    robert_qa_pairs = generate_qa_pairs(robert_file_path, persona_attributes, model)

    # Generate QA pairs for Dr. Sanjay Gupta
    sanjay_qa_pairs = generate_qa_pairs(sanjay_file_path, persona_attributes, model)

    # Concatenate QA pairs for both personas
    qa_pairs = robert_qa_pairs + sanjay_qa_pairs
    # Define the file path for saving the JSON file
    output_file_path = "C:/Project_Data/qa_generated_data/persona_qa_pairs.json"

    # Save the combined QA pairs to a JSON file
    with open(output_file_path, "w") as json_file:
        json.dump(qa_pairs, json_file)

    print("Combined QA pairs saved to:", output_file_path)

if __name__ == "__main__":
    main()
