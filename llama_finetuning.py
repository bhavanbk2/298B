#from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb, platform, gradio, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login
import pandas as pd

dataset = [
        {"persona": "Robert Kiyosaki", "interview": [
            {"question": "Robert Kiyosaki, welcome to So Money. It is an honor and a pleasure to have you on the show.", "answer": "Oh, thank you. Thank you."},
            {"question": "Well, the new book is called Second Chance and it’s really about one of my mentors. I’ve had many mentors and one of my mentors is a man named Dr. R. Buckminster Fuller. He’s most wellknown possibly for the geodesic dome. But he was known as a futurist. He could see, they call him the man who could see the future but he’s the first guy that kind of systematized how a person sees the future. So, I began studying with Fuller back in 1967, if you can imagine that, and I hitchhiked from New York City when I was in school and I hitchhiked to Montreal, Canada to see Expo ‘67 but more importantly I wanted to see Bucky’s geodesic dome which was a U.S. pavilion at the world’s fair in Montreal. And, the Montreal World’s Fair of ’67 was the exposition on the future so I wanted to see the future. And, over the time I have had the pleasure and the benefit of studying with Fuller 3 times and each time I could better see the future.", "answer": "Well, the new book is called Second Chance and it’s really about one of my mentors. I’ve had many mentors and one of my mentors is a man named Dr. R. Buckminster Fuller. He’s most wellknown possibly for the geodesic dome. But he was known as a futurist. He could see, they call him the man who could see the future but he’s the first guy that kind of systematized how a person sees the future. So, I began studying with Fuller back in 1967, if you can imagine that, and I hitchhiked from New York City when I was in school and I hitchhiked to Montreal, Canada to see Expo ‘67 but more importantly I wanted to see Bucky’s geodesic dome which was a U.S. pavilion at the world’s fair in Montreal. And, the Montreal World’s Fair of ’67 was the exposition on the future so I wanted to see the future. And, over the time I have had the pleasure and the benefit of studying with Fuller 3 times and each time I could better see the future."},
            {"question": "What is the future of money in 2015 and in the future?", "answer": "Well, I’m let me say this much, on the big picture the rich as you know are getting richer and unfortunately the poorer and middle class are getting poorer. And, that’s why I wrote Rich Dad Poor Dad and in Rich Dad Poor Dad I said, “You savers are losers. Your house is not an asset and the rich don’t work for money.” And, what I was doing was preparing people for now, this is the future. So, as you know, the book Rich Dad Poor Dad came out in 1997 and I said, “Your house is not an asset” and I was trashed by almost every financial person. “How can you say that, you know?” And then, what happens in 2007, the world found out that your house is not an asset when their subprime mortgage crashed."},
            {"question": "Do you still think we could see a crash in 2016? ", "answer": "I wouldn’t have said it unless I thought it was possible. And, it’s not a stock market and this is the point, okay, this is not a Stock market crash.I wouldn’t have said it unless I thought it was possible. And, it’s not a stock market and this is the point, okay, this is not a Stock market crash."},
        ]}
    ]
# Read Excel file
df = pd.read_excel(r"C:\Users\shash\Downloads\archive\QA.xlsx")

# Initialize the dataset list
dataset = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Extract persona, question, and answer from the row
    persona = row['Persona']
    question = row['question']
    answer = row['answers']
    
    # Create a dictionary for the interview
    interview = {
        "question": question,
        "answer": answer
    }
    
    # Check if the persona already exists in the dataset
    persona_exists = False
    for item in dataset:
        if item["persona"] == persona:
            item["interview"].append(interview)
            persona_exists = True
            break
    
    # If the persona doesn't exist, create a new entry for it
    if not persona_exists:
        dataset.append({
            "persona": persona,
            "interview": [interview]
        })


from collections import defaultdict
from sklearn.model_selection import train_test_split

# Group the dataset by persona
persona_data = defaultdict(list)
for item in dataset:
    persona_data[item["persona"]].append(item["interview"])

# Initialize empty dictionaries to store splits
train_data = defaultdict(list)
val_data = defaultdict(list)
test_data = defaultdict(list)

# Split each persona's interviews into training, validation, and test sets
for persona, interviews in persona_data.items():
    train_interviews, other_interviews = train_test_split(interviews[0], test_size=0.2, random_state=42)
    val_interviews, test_interviews = train_test_split(other_interviews, test_size=0.5, random_state=42)
    
    train_data[persona] = train_interviews
    val_data[persona] = val_interviews
    test_data[persona] = test_interviews

# Print the sizes of the splits for each persona
for persona in persona_data.keys():
    print(f"Persona: {persona}")
    print("Training data size:", len(train_data[persona]))
    print("Validation data size:", len(val_data[persona]))
    print("Test data size:", len(test_data[persona]))
    print()

# Pre trained model
model_name = "meta-llama/Llama-2-7b-hf"

# Dataset name
dataset_name = "vicgalle/alpaca-gpt4"

# Hugging face repository link to save fine-tuned model(Create new repository in huggingface,copy and paste here)

new_model = "chatbot/llama-2-7b-sjsu-msda"
hugging_face_token = os.getenv("Hugging_face_token")
# Load base model(llama-2-7b-hf) and tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.float16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=bnb_config,
    revision="main",
    auth_token=hugging_face_token
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
'''
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, auth_token=hugging_face_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

#monitering login
wandb.login(key=os.getenv("WANDB_API"))
run = wandb.init(project='llama-2-7B Fine tuning', job_type="training", anonymous="allow", auth_token=hugging_face_token)


peft_config = LoraConfig(
    lora_alpha= 8,
    lora_dropout= 0.1,
    r= 16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
)

training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 20,
    per_device_train_batch_size= 6,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 1000,
    logging_steps= 20,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "linear",
    report_to="wandb"
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

# Train model
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()

model.push_to_hub(new_model)
tokenizer.push_to_hub(new_model)
'''