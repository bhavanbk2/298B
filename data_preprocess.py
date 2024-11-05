import pdfreader as pdf
import os

def save_as_txt(file_path,data):
    # Open the file in append mode and append the string data to it
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write("\n")  # Add a new line before appending
        file.write(data)

def process_domain(base_dir):
    # Dictionary to store cleaned text by domain
    cleaned_text_by_domain = {}
    medical_domain_data = []
    finance_domain_data = {}

    # Process domain data (Finance and Medical)
    domain_mapping = {
        "Finance": 7,  # Number of PDFs in the Finance domain
        "Medical": 1   # Number of PDFs in the Medical domain
    }
    
    for domain, num_pdfs in domain_mapping.items():
        domain_dir = os.path.join(base_dir, "Domain", domain)
        
        # List to store cleaned text for each PDF in the domain
        cleaned_text_list = []
        
        # Process each PDF in the domain directory
        for pdf_file in os.listdir(domain_dir)[:num_pdfs]:
            pdf_file_path = os.path.join(domain_dir, pdf_file)
            extracted_text = pdf.extract_text_from_pdf(pdf_file_path)

            if extracted_text:
                cleaned_text = pdf.clean_text(extracted_text) #Data Cleansing
                cleaned_text = pdf.replace_contractions(cleaned_text) #Data Preprocessing
                cleaned_text = pdf.advanced_text_processing(cleaned_text) #Advanced Processing
                cleaned_text_list.append(cleaned_text)
        
        # Store cleaned text list in dictionary with domain as key
        cleaned_text_by_domain[domain] = cleaned_text_list
        
        # Concatenate cleaned text for Finance and Medical domains
        if domain == "Finance":
            finance_domain_data = cleaned_text_list
        elif domain == "Medical":
            medical_domain_data = cleaned_text_list
        
        # # Save cleaned text to a text file
        # concatenated_string = ''.join(cleaned_text_list)
        # save_as_txt(os.path.join(base_dir, 'cleansed_data', f'{domain.lower()}_domain.txt'), concatenated_string)

    return cleaned_text_by_domain, medical_domain_data, finance_domain_data

def process_persona(base_dir):
    # Dictionary to store cleaned text by personality
    cleaned_text_by_personality = {}
    robert_persona_data = []
    sanjay_persona_data = []

    # Process personality data (Robert and Sanjay)
    personality_mapping = {
        "Robert": 5,   # Number of PDFs for Robert's personality
        "Sanjay": 2    # Number of PDFs for Sanjay's personality
    }

    for personality, num_pdfs in personality_mapping.items():
        personality_dir = os.path.join(base_dir, "Personality", personality)
        
        # List to store cleaned text for each PDF in the personality
        cleaned_text_list = []
            
        # Process each PDF in the personality directory
        for pdf_file in os.listdir(personality_dir)[:num_pdfs]:
            pdf_file_path = os.path.join(personality_dir, pdf_file)
            extracted_text = pdf.extract_text_from_pdf(pdf_file_path)

            if extracted_text:
                cleaned_text = pdf.clean_text(extracted_text) #Data Cleansing
                cleaned_text = pdf.replace_contractions(cleaned_text) #Data Preprocessing
                cleaned_text = pdf.advanced_text_processing(cleaned_text) #Advanced Processing
                cleaned_text_list.append(cleaned_text)
            
        # Store cleaned text list in dictionary with personality as key
        cleaned_text_by_personality[personality] = cleaned_text_list
        
        # Assign cleaned text to Robert and Sanjay persona data
        if personality == "Robert":
            robert_persona_data = cleaned_text_list
        elif personality == "Sanjay":
            sanjay_persona_data = cleaned_text_list
        
        # # Save cleaned text to a text file
        # concatenated_string = ''.join(cleaned_text_list)
        # save_as_txt(os.path.join(base_dir, 'cleansed_data', f'{personality.lower()}_persona.txt'), concatenated_string)

    return cleaned_text_by_personality, robert_persona_data, sanjay_persona_data


# This is how you get data

# base_dir = "C:/Project_Data"

# # Process both domain and personality data using the functions defined
# domain_data, medical_domain_data, finance_domain_data = process_domain(base_dir)
# persona_data, robert_persona_data, sanjay_persona_data = process_persona(base_dir)
