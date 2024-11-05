from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

def segment_into_qa_pairs():
    # Segment the transcribed text into QA pairs (not shown, could use NLP techniques like sentence splitting)
    persona_qa_data = [
        {"persona": "Robert Kiyosaki", "interview": [
            {"question": "Robert	Kiyosaki,	welcome	to	So	Money.	It	is	an	honor	and	a	pleasure	to	have	you	on	the	show.", "answer": "Oh,	thank	you.	Thank	you."},
            {"question": "Well,	the	new	book	is	called Second	Chance and	it’s	really	about	one	of	my	mentors.	I’ve	had	many	mentors	and	one of	my	mentors	is	a	man	named	Dr.	R.	Buckminster	Fuller.	He’s	most	wellknown	possibly	for	the	geodesic	dome. But	he	was	known	as	a	futurist.	He	could	see,	they	call	him	the	man	who	could	see	the	future	but	he’s	the	first	guy	that	kind	of	systematized	how	a	person	sees	the	future.	So,	I	began	studying	with	Fuller	back	in	1967,	if	you	can	imagine	that,	and	I	hitchhiked	from	New	York	City	when	I	was	in	school and	I	hitchhiked	to	Montreal,	Canada	to	see	Expo	‘67	but	more	importantly	I	wanted	to	see	Bucky’s	geodesic	dome	which	was	a	U.S.	pavilion	at	the	world’s	fair	in	Montreal. And,	the	Montreal	World’s	Fair	of	’67	was	the	exposition	on	the	future	so	I	wanted	to	see	the	future.	And,	over	the	time	I	have	had	the	pleasure	and	the	benefit	of	studying	with	Fuller	3	times	and	each	time	I	could	better	see	the	future.", "answer": "Well,	the	new	book	is	called Second	Chance and	it’s	really	about	one	of	my	mentors.	I’ve	had	many	mentors	and	one of	my	mentors	is	a	man	named	Dr.	R.	Buckminster	Fuller.	He’s	most	wellknown	possibly	for	the	geodesic	dome. But	he	was	known	as	a	futurist.	He	could	see,	they	call	him	the	man	who	could	see	the	future	but	he’s	the	first	guy	that	kind	of	systematized	how	a	person	sees	the	future.	So,	I	began	studying	with	Fuller	back	in	1967,	if	you	can	imagine	that,	and	I	hitchhiked	from	New	York	City	when	I	was	in	school and	I	hitchhiked	to	Montreal,	Canada	to	see	Expo	‘67	but	more	importantly	I	wanted	to	see	Bucky’s	geodesic	dome	which	was	a	U.S.	pavilion	at	the	world’s	fair	in	Montreal. And,	the	Montreal	World’s	Fair	of	’67	was	the	exposition	on	the	future	so	I	wanted	to	see	the	future.	And,	over	the	time	I	have	had	the	pleasure	and	the	benefit	of	studying	with	Fuller	3	times	and	each	time	I	could	better	see	the	future."},
            {"question": "What	is	the	future	of	money	in	2015	and	in	the	future?", "answer": "Well,	I’m	let	me say	this	much,	on	the	big	picture	the	rich	as	you	know	are	getting	richer	and	unfortunately	the	poorer	and	middle	class	are	getting	poorer.	And,	that’s	why	I	wrote	Rich	Dad	Poor	Dad and	in	Rich	Dad	Poor	Dad I	said,	“You	savers	are	losers.	Your	house	is	not	an	asset	and	the	rich	don’t	work	for	money.”	And,	what	I	was	doing was	preparing	people for	now,	this	is	the	future.	So,	as	you	know,	the	book	Rich	Dad	Poor	Dad came	out	in	1997	and	I	said,	“Your	house	is	not	an	asset”	and	I	was	trashed	by	almost	every	financial person.	“How	can	you	say	that,	you	know?”	And	then,	what	happens	in	2007,	the	world	found	out	that	your	house	is	not	an	asset	when	their	subprime	mortgage	crashed."},
            {"question": "Do	you	still	think	we	could	see	a	crash	in	2016?	", "answer": "I	wouldn’t	have	said	it	unless	I	thought	it	was	possible.	And,	it’s	not	a	stock	market	and	this	is	the	point,	okay,	this	is	not	a	Stock	market	crash.I	wouldn’t	have	said	it	unless	I	thought	it	was	possible.	And,	it’s	not	a	stock	market	and	this	is	the	point,	okay,	this	is	not	a	Stock	market	crash."},
        ]}
    ]
    return persona_qa_data
# Function to tokenize and fine-tune the GPT-2 model
def tokenize_and_finetune_gpt2(persona_qa_data):
    # Tokenize persona-QA pairs
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token

    inputs = [f"{p['persona']} <|endoftext|> Q: {q['question']} A: {q['answer']}" for p in persona_qa_data for q in p["interview"]]
    encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    # Convert the list of input sequences to a flat list of input IDs
    input_ids = [input_id for input_id_list in encoded_inputs["input_ids"] for input_id in input_id_list]

    # Fine-tune the pre-trained GPT-2 model on persona-QA pairs
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    

    # Define a custom dataset class to work with tensors
    class TensorDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create a dataset from encoded inputs
    train_dataset = TensorDataset(encoded_inputs["input_ids"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Define the output directory where the trained model will be saved
    output_dir = "./output"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Output directory for saving the model
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_gpt2_persona_qa")

persona_qa_data = segment_into_qa_pairs()
tokenize_and_finetune_gpt2(persona_qa_data)
