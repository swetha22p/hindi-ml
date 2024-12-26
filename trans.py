# from flask import Flask, request, jsonify
# import os,re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
# # from langchain.llms import HuggingFacePipeline
# from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})



# # Check and Login to Hugging Face Hub
# def check_huggingface_login():
#     """
#     Check Hugging Face CLI login status. If not logged in, prompt for token.
#     """
#     try:
#         # Check if logged in
#         os.system("huggingface-cli whoami")
#         print("Hugging Face CLI is already logged in.")
#     except Exception:
#         # Prompt user to log in if not logged in
#         print("Hugging Face CLI not logged in. Please provide your Hugging Face token.")
#         os.system("huggingface-cli login")

# # Load Pretrained Model and Tokenizer
# def load_model_and_tokenizer():
#     # Ensure Hugging Face is logged in
#     check_huggingface_login()

#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#     model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         model.resize_token_embeddings(len(tokenizer))

#     return model, tokenizer

# # Function to Read Dataset
# def read_dataset(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if "\t" in line:  # Ensure it has the tab separator
#                 linearized_input, natural_sentence = line.strip().split("\t")
#                 data.append({"input": linearized_input, "output": natural_sentence})
#     return data

# # Prepare Few-Shot Examples
# def prepare_few_shot_prompt(data, num_examples=130):
#     examples = data[:num_examples]

#     # Create an example prompt template
#     example_prompt = ChatPromptTemplate.from_messages(
#         [("human", "{input}"), ("ai", "{output}")]
#     )

#     # Create a few-shot prompt template
#     few_shot_prompt = FewShotChatMessagePromptTemplate(
#         examples=examples,
#         example_prompt=example_prompt,
#     )

#     # Combine into a final prompt template
#     final_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "Above are some examples of linearised graphs and their corresponding sentences in English. Based on the examples provided above please generate full sentences in English from the linearised graph provided below as query Generate only the sentence in English as the answer. Do not repeat the query graph in the answer. Do not repeat the examples in the answer."),
#             few_shot_prompt,
#             ("human", "{input}"),
#         ]
#     )
#     return final_prompt

# # Train Few-Shot Model
# def train_few_shot_model(model, tokenizer, data, num_examples=130):
#     # Prepare few-shot examples
#     prompt_template = prepare_few_shot_prompt(data, num_examples)

#     # Create a pipeline for text generation
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         device=0 if torch.cuda.is_available() else -1,
#         max_new_tokens=50,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.9
#     )

#     # Wrap the pipeline with LangChain's HuggingFacePipeline
#     llm = HuggingFacePipeline(pipeline=pipe)

#     return prompt_template, model, tokenizer

# # Predict Sentence from Input
# def predict_sentence(input_text, model, tokenizer, prompt_template):
#     # Format the input with the prompt
#     formatted_prompt = prompt_template.format(input=input_text)

#     # Tokenize the input prompt
#     inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

#     # Generate the response
#     output = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 50, temperature=0.7, top_k=50, top_p=0.9)

#     # Decode the output
#     decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Extract and return the final response
#     final_response = decoded_output.split("AI:")[-1].strip()  # Adjust based on output format
#     return final_response



# # Initialize the model and dataset
# model, tokenizer = load_model_and_tokenizer()
# file_path = "/home/praveen/Desktop/ml_based/mlbased/131_data.txt"
# data = read_dataset(file_path)
# prompt_template, model, tokenizer = train_few_shot_model(model, tokenizer, data)

# # Flask API endpoint
# @app.route('/generate', methods=['POST'])
# def generate_sentence():
#     req_data = request.json
#     input_text = req_data.get('input_text', '')
#     if not input_text:
#         return jsonify({'error': 'Input text is required'}), 400
#     print(input_text)

#     try:
#         result = predict_sentence(input_text, model, tokenizer, prompt_template)
#         return jsonify({'result': resul
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)


from flask import Flask, request, jsonify
import os,re
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from flask_cors import CORS

app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": "http://localhost:43821"}})
CORS(app, resources={r"/generate": {"origins": "*"}})





# Load Pretrained Model and Tokenizer
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/home/user/hindiml_based/lamma_model/llama3.2-1b")
    model = AutoModelForCausalLM.from_pretrained("/home/user/hindiml_based/lamma_model/llama3.2-1b")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
# Read Dataset
def read_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "\t" in line:
                linearized_input, natural_sentence = line.strip().split("\t")
                data.append({"input": linearized_input, "output": natural_sentence})
    return data

# Prepare Few-Shot Examples
def prepare_few_shot_prompt(data, num_examples=130):
    examples = data[:num_examples]
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate sentences based on the examples."),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt
def train_few_shot_model(model, tokenizer, data, num_examples=130):
    prompt_template = prepare_few_shot_prompt(data, num_examples)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return prompt_template, model, tokenizer

# Predict Sentence
def predict_sentence(input_text, model, tokenizer, prompt_template):
    formatted_prompt = prompt_template.format(input=input_text)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 50, temperature=0.7, top_k=50, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    final_response = decoded_output.split("AI:")[-1].strip()
    cleaned_text = re.sub(r"Human:.*", "", final_response, flags=re.DOTALL).strip()
    print(f"Model Response: {cleaned_text}\n")
    return cleaned_text

model, tokenizer = load_model_and_tokenizer()
file_path = "/home/user/hindiml_based/lamma_model/131_data.txt"
data = read_dataset(file_path)
prompt_template, model, tokenizer = train_few_shot_model(model, tokenizer, data)

# Flask API endpoint
@app.route('/generate', methods=['POST'])
def generate_sentence():
    req_data = request.json
    input_text = req_data.get('input_text', '')
    if not input_text:
        return jsonify({'error': 'Input text is required'}), 400
    print(input_text)

    try:
        result = predict_sentence(input_text, model, tokenizer, prompt_template)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
