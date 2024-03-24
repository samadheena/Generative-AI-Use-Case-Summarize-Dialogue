Generative AI Use Case: Summarize Dialogue
Welcome to the practical side of this course. In this lab you will do the dialogue summarization task using generative AI. You will explore how the input text affects the output of the model, and perform prompt engineering to direct it towards the task you need. By comparing zero shot, one shot, and few shot inferences, you will take the first step towards prompt engineering and see how it can enhance the generative output of Large Language Models.

Table of Contents
1 - Set up Kernel and Required Dependencies
2 - Summarize Dialogue without Prompt Engineering
3 - Summarize Dialogue with an Instruction Prompt
3.1 - Zero Shot Inference with an Instruction Prompt
3.2 - Zero Shot Inference with the Prompt Template from FLAN-T5
4 - Summarize Dialogue with One Shot and Few Shot Inference
4.1 - One Shot Inference
4.2 - Few Shot Inference
5 - Generative Configuration Parameters for Inference

1 - Set up Kernel and Required Dependencies
First, check that the correct kernel is chosen.

Image
You can click on that (top right of the screen) to see and check the details of the image, kernel, and instance type.

ImageTime alert close


import os

instance_type_expected = 'ml-m5-2xlarge'
instance_type_current = os.environ.get('HOSTNAME')

print(f'Expected instance type: instance-datascience-{instance_type_expected}')
print(f'Currently chosen instance type: {instance_type_current}')

assert instance_type_expected in instance_type_current, f'ERROR. You selected the {instance_type_current} instance type. Please select {instance_type_expected} instead as shown on the screenshot above'
print("Instance type has been chosen correctly.")


Now install the required packages to use PyTorch and Hugging Face transformers and datasets.
%pip install -U datasets==2.17.0

%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

%pip install \
    transformers==4.27.2 --quiet







    Load the datasets, Large Language Model (LLM), tokenizer, and configurator. Do not worry if you do not understand yet all of those components - they will be described and discussed later in the notebook.

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig



2 - Summarize Dialogue without Prompt Engineering
In this use case, you will be generating a summary of a dialogue with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face. The list of available models in the Hugging Face transformers package can be found here.

Let's upload some simple dialogues from the DialogSum Hugging Face dataset. This dataset contains 10,000+ dialogues with the corresponding manually labeled summaries and topics.



huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)


Print a couple of dialogues with their baseline summaries.
example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print('INPUT DIALOGUE:')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY:')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print() 



    Load the FLAN-T5 model, creating an instance of the AutoModelForSeq2SeqLM class with the .from_pretrained() method.
To perform encoding and decoding, you need to work with text in a tokenized form. Tokenization is the process of splitting texts into smaller units that can be processed by the LLM models.

Download the tokenizer for the FLAN-T5 model using AutoTokenizer.from_pretrained() method. Parameter use_fast switches on fast tokenizer. At this stage, there is no need to go into the details of that, but you can find the tokenizer parameters in the documentation.

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

Test the tokenizer encoding and decoding a simple sentence:
sentence = "What time is it, Tom?"

sentence_encoded = tokenizer(sentence, return_tensors='pt')

sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0], 
        skip_special_tokens=True
    )

print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print('\nDECODED SENTENCE:')
print(sentence_decoded)


    Now it's time to explore how well the base LLM summarizes a dialogue without any prompt engineering. **Prompt engineering** is an act of a human changing the **prompt** (input) to improve the response for a given task.
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')






3 - Summarize Dialogue with an Instruction Prompt
Prompt engineering is an important concept in using foundation models for text generation. You can check out this blog from Amazon Science for a quick introduction to prompt engineering.


3.1 - Zero Shot Inference with an Instruction Prompt
In order to instruct the model to perform a task - summarize a dialogue - you can take the dialogue and convert it into an instruction prompt. This is often called zero shot inference. You can check out this blog from AWS for a quick description of what zero shot learning is and why it is an important concept to the LLM model.

Wrap the dialogue in a descriptive instruction and see how the generated text will change:


This is much better! But the model still does not pick up on the nuance of the conversations though.

Exercise:

Experiment with the prompt text and see how the inferences will be changed. Will the inferences change if you end the prompt with just empty string vs. Summary: ?
Try to rephrase the beginning of the prompt text from Summarize the following conversation. to something different - and see how it will influence the generated output.

3.2 - Zero Shot Inference with the Prompt Template from FLAN-T5
Let's use a slightly different prompt. FLAN-T5 has many prompt templates that are published for certain tasks here. In the following code, you will use one of the pre-built FLAN-T5 prompts:





4 - Summarize Dialogue with One Shot and Few Shot Inference
One shot and few shot inference are the practices of providing an LLM with either one or more full examples of prompt-response pairs that match your task - before your actual prompt that you want completed. This is called "in-context learning" and puts your model into a state that understands your specific task. You can read more about it in this blog from HuggingFace.


4.1 - One Shot Inference
Let's build a function that takes a list of example_indices_full, generates a prompt with full examples, then at the end appends the prompt which you want the model to complete (example_index_to_summarize). You will use the same FLAN-T5 prompt template from section 3.2.
Construct the prompt to perform one shot inference:
4.2 - Few Shot Inference
Let's explore few shot inference by adding two more full dialogue-summary pairs to your prompt.



Token indices sequence length is longer than the specified maximum sequence length for this model (819 > 512). Running this sequence through the model will result in indexing errors
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

---------------------------------------------------------------------------------------------------
MODEL GENERATION - FEW SHOT:
#Person1 wants to upgrade his system. #Person2 wants to add a painting program to his software. #Person1 wants to upgrade his hardware.
In this case, few shot did not provide much of an improvement over one shot inference. And, anything above 5 or 6 shot will typically not help much, either. Also, you need to make sure that you do not exceed the model's input-context length which, in our case, if 512 tokens. Anything above the context length will be ignored.

However, you can see that feeding in at least one full example (one shot) provides the model with more information and qualitatively improves the summary overall.

Exercise:

Experiment with the few shot inferencing.

Choose different dialogues - change the indices in the example_indices_full list and example_index_to_summarize value.
Change the number of shots. Be sure to stay within the model's 512 context length, however.
How well does few shot inferencing work with other examples?



5 - Generative Configuration Parameters for Inference
You can change the configuration parameters of the generate() method to see a different output from the LLM. So far the only parameter that you have been setting was max_new_tokens=50, which defines the maximum number of tokens to generate. A full list of available parameters can be found in the Hugging Face Generation documentation.

A convenient way of organizing the configuration parameters is to use GenerationConfig class.

Exercise:

Change the configuration parameters to investigate their influence on the output.

Putting the parameter do_sample = True, you activate various decoding strategies which influence the next token from the probability distribution over the entire vocabulary. You can then adjust the outputs changing temperature and other parameters (such as top_k and top_p).

Uncomment the lines in the cell below and rerun the code. Try to analyze the results. You can read some comments below.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - FEW SHOT:
#Person1 wants to upgrade his system. #Person2 wants to add a painting program to his software. #Person1 wants to upgrade his hardware.
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

Comments related to the choice of the parameters in the code cell above:

Choosing max_new_tokens=10 will make the output text too short, so the dialogue summary will be cut.
Putting do_sample = True and changing the temperature value you get more flexibility in the output.
As you can see, prompt engineering can take you a long way for this use case, but there are some limitations. Next, you will start to explore how you can use fine-tuning to help your LLM to understand a particular use case in better depth!







