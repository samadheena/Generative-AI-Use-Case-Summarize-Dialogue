# Generative-AI-Use-Case-Summarize-Dialogue{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Generative AI Use Case: Summarize Dialogue\n",
    "\n",
    "Welcome to the practical side of this course. In this lab you will do the dialogue summarization task using generative AI. You will explore how the input text affects the output of the model, and perform prompt engineering to direct it towards the task you need. By comparing zero shot, one shot, and few shot inferences, you will take the first step towards prompt engineering and see how it can enhance the generative output of Large Language Models."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Table of Contents"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- [ 1 - Set up Kernel and Required Dependencies](#1)\n",
    "- [ 2 - Summarize Dialogue without Prompt Engineering](#2)\n",
    "- [ 3 - Summarize Dialogue with an Instruction Prompt](#3)\n",
    "  - [ 3.1 - Zero Shot Inference with an Instruction Prompt](#3.1)\n",
    "  - [ 3.2 - Zero Shot Inference with the Prompt Template from FLAN-T5](#3.2)\n",
    "- [ 4 - Summarize Dialogue with One Shot and Few Shot Inference](#4)\n",
    "  - [ 4.1 - One Shot Inference](#4.1)\n",
    "  - [ 4.2 - Few Shot Inference](#4.2)\n",
    "- [ 5 - Generative Configuration Parameters for Inference](#5)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a name='1'></a>\n",
    "## 1 - Set up Kernel and Required Dependencies"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "First, check that the correct kernel is chosen.\n",
    "\n",
    "<img src=\"images/kernel_set_up.png\" width=\"300\"/>\n",
    "\n",
    "You can click on that (top right of the screen) to see and check the details of the image, kernel, and instance type.\n",
    "\n",
    "<img src=\"images/w1_kernel_and_instance_type.png\" width=\"600\"/>\n",
    "\n",
    "<img src=\"data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxMjUiIHZpZXdCb3g9IjAgMCA4MDAgMTI1IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJmYWRlR3JhZGllbnQiIHgxPSIwIiB4Mj0iMSI+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNGMEYwRjAiLz4KICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgICAgICA8L2xpbmVhckdyYWRpZW50PgogICAgICAgIDxtYXNrIGlkPSJmYWRlTWFzayI+CiAgICAgICAgICAgIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSI3NTAiIGhlaWdodD0iMTI1IiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSIxMjUiIGZpbGw9InVybCgjZmFkZUdyYWRpZW50KSIvPgogICAgICAgIDwvbWFzaz4KICAgIDwvZGVmcz4KICAgIDxwYXRoIGQ9Ik01MCw5NyBBNTAsNTAgMCAwIDEgNTMsMyBMNzk3LCAzIEw3OTcsOTcgTDUwLDk3IFoiIGZpbGw9IiNGMEYwRjAiIHN0cm9rZT0iI0UwRTBFMCIgc3Ryb2tlLXdpZHRoPSIxIiBtYXNrPSJ1cmwoI2ZhZGVNYXNrKSIvPgogICAgPHRleHQgeD0iMTAwIiB5PSIzNCIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5QbGVhc2UgbWFrZSBzdXJlIHRoYXQgeW91IGNob29zZTwvdGV4dD4gCiAgICA8dGV4dCB4PSIzMjAiIHk9IjM0IiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiMzMzMzMzMiIGZvbnQtd2VpZ2h0PSJib2xkIj5tbC5tNS4yeGxhcmdlPC90ZXh0PgogICAgPHRleHQgeD0iNDE4IiB5PSIzNCIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5pbnN0YW5jZSB0eXBlLjwvdGV4dD4KICAgIDx0ZXh0IHg9IjEwMCIgeT0iNTYiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzMzMzMzMyI+VG8gZmluZCB0aGF0IGluc3RhbmNlIHR5cGUsIHlvdSBtaWdodCBoYXZlIHRvIHNjcm9sbCBkb3duIHRvIHRoZSAiQWxsIEluc3RhbmNlcyIgc2VjdGlvbiBpbiB0aGUgZHJvcGRvd24uPC90ZXh0PgogICAgPHRleHQgeD0iMTAwIiB5PSI3OCIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5DaG9pY2Ugb2YgYW5vdGhlciBpbnN0YW5jZSB0eXBlIG1pZ2h0IGNhdXNlIHRyYWluaW5nIGZhaWx1cmUva2VybmVsIGhhbHQvYWNjb3VudCBkZWFjdGl2YXRpb24uPC90ZXh0Pgo8L3N2Zz4K\" alt=\"Time alert close\"/>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "instance_type_expected = 'ml-m5-2xlarge'\n",
    "instance_type_current = os.environ.get('HOSTNAME')\n",
    "\n",
    "print(f'Expected instance type: instance-datascience-{instance_type_expected}')\n",
    "print(f'Currently chosen instance type: {instance_type_current}')\n",
    "\n",
    "assert instance_type_expected in instance_type_current, f'ERROR. You selected the {instance_type_current} instance type. Please select {instance_type_expected} instead as shown on the screenshot above'\n",
    "print(\"Instance type has been chosen correctly.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "Now install the required packages to use PyTorch and Hugging Face transformers and datasets.\n",
    "\n",
    "<img src=\"data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxMjUiIHZpZXdCb3g9IjAgMCA4MDAgMTI1IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJmYWRlR3JhZGllbnQiIHgxPSIwIiB4Mj0iMSI+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNGMEYwRjAiLz4KICAgICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgICAgICA8L2xpbmVhckdyYWRpZW50PgogICAgICAgIDxtYXNrIGlkPSJmYWRlTWFzayI+CiAgICAgICAgICAgIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSI3NTAiIGhlaWdodD0iMTI1IiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSIxMjUiIGZpbGw9InVybCgjZmFkZUdyYWRpZW50KSIvPgogICAgICAgIDwvbWFzaz4KICAgIDwvZGVmcz4KICAgIDxwYXRoIGQ9Ik0zLDUwIEE1MCw1MCAwIDAgMSA1MywzIEw3OTcsMyBMNzk3LDk3IEw5Nyw5NyBMNTAsMTE1IEwzLDk3IFoiIGZpbGw9IiNGMEYwRjAiIHN0cm9rZT0iI0UwRTBFMCIgc3Ryb2tlLXdpZHRoPSIxIiBtYXNrPSJ1cmwoI2ZhZGVNYXNrKSIvPgogICAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iMzAiIGZpbGw9IiM1N2M0ZjgiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgICA8Y2lyY2xlIGN4PSI1MCIgY3k9IjUwIiByPSIyNSIgZmlsbD0iI0YwRjBGMCIvPgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSI1MCIgeTI9IjMwIiBzdHJva2U9IiM1N2M0ZjgiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CiAgICA8bGluZSB4MT0iNTAiIHkxPSI1MCIgeDI9IjY1IiB5Mj0iNTAiIHN0cm9rZT0iIzU3YzRmOCIgc3Ryb2tlLXdpZHRoPSIzIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KICAgIDx0ZXh0IHg9IjEwMCIgeT0iMzQiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzMzMzMzMyI+VGhlIG5leHQgY2VsbCBtYXkgdGFrZSBhIGZldyBtaW51dGVzIHRvIHJ1bi4gUGxlYXNlIGJlIHBhdGllbnQuPC90ZXh0PgogICAgPHRleHQgeD0iMTAwIiB5PSI1NiIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjMzMzMzMzIj5JZ25vcmUgdGhlIHdhcm5pbmdzIGFuZCBlcnJvcnMsIGFsb25nIHdpdGggdGhlIG5vdGUgYWJvdXQgcmVzdGFydGluZyB0aGUga2VybmVsIGF0IHRoZSBlbmQuPC90ZXh0Pgo8L3N2Zz4K\" alt=\"Time alert open medium\"/>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "%pip install -U datasets==2.17.0\n",
    "\n",
    "%pip install --upgrade pip\n",
    "%pip install --disable-pip-version-check \\\n",
    "    torch==1.13.1 \\\n",
    "    torchdata==0.5.1 --quiet\n",
    "\n",
    "%pip install \\\n",
    "    transformers==4.27.2 --quiet"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "<img src=\"data:image/svg+xml;base64,Cjxzdmcgd2lkdGg9IjgwMCIgaGVpZ2h0PSI1MCIgdmlld0JveD0iMCAwIDgwMCA1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxkZWZzPgogICAgICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZmFkZUdyYWRpZW50IiB4MT0iMCIgeDI9IjEiPgogICAgICAgICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjRjBGMEYwIi8+CiAgICAgICAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI0YwRjBGMCIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgICA8bWFzayBpZD0iZmFkZU1hc2siPgogICAgICAgICAgICA8cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iNzUwIiBoZWlnaHQ9IjUwIiBmaWxsPSJ3aGl0ZSIvPgogICAgICAgICAgICA8cmVjdCB4PSI3NTAiIHk9IjAiIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCIgZmlsbD0idXJsKCNmYWRlR3JhZGllbnQpIi8+CiAgICAgICAgPC9tYXNrPgogICAgPC9kZWZzPgogICAgPHBhdGggZD0iTTI1LDUwIFEwLDUwIDAsMjUgTDUwLDMgTDk3LDI1IEw3OTcsMjUgTDc5Nyw1MCBMMjUsNTAgWiIgZmlsbD0iI0YwRjBGMCIgc3Ryb2tlPSIjRTBFMEUwIiBzdHJva2Utd2lkdGg9IjEiIG1hc2s9InVybCgjZmFkZU1hc2spIi8+Cjwvc3ZnPgo=\" alt=\"Time alert close\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "Load the datasets, Large Language Model (LLM), tokenizer, and configurator. Do not worry if you do not understand yet all of those components - they will be described and discussed later in the notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from datasets import load_dataset\n",
    "from transformers import AutoModelForSeq2SeqLM\n",
    "from transformers import AutoTokenizer\n",
    "from transformers import GenerationConfig"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a name='2'></a>\n",
    "## 2 - Summarize Dialogue without Prompt Engineering\n",
    "\n",
    "In this use case, you will be generating a summary of a dialogue with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face. The list of available models in the Hugging Face `transformers` package can be found [here](https://huggingface.co/docs/transformers/index). \n",
    "\n",
    "Let's upload some simple dialogues from the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) Hugging Face dataset. This dataset contains 10,000+ dialogues with the corresponding manually labeled summaries and topics. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "huggingface_dataset_name = \"knkarthick/dialogsum\"\n",
    "\n",
    "dataset = load_dataset(huggingface_dataset_name)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "Print a couple of dialogues with their baseline summaries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "example_indices = [40, 200]\n",
    "\n",
    "dash_line = '-'.join('' for x in range(100))\n",
    "\n",
    "for i, index in enumerate(example_indices):\n",
    "    print(dash_line)\n",
    "    print('Example ', i + 1)\n",
    "    print(dash_line)\n",
    "    print('INPUT DIALOGUE:')\n",
    "    print(dataset['test'][index]['dialogue'])\n",
    "    print(dash_line)\n",
    "    print('BASELINE HUMAN SUMMARY:')\n",
    "    print(dataset['test'][index]['summary'])\n",
    "    print(dash_line)\n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Load the [FLAN-T5 model](https://huggingface.co/docs/transformers/model_doc/flan-t5), creating an instance of the `AutoModelForSeq2SeqLM` class with the `.from_pretrained()` method. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "iAYlS40Z3l-v",
    "tags": []
   },
   "outputs": [],
   "source": [
    "model_name='google/flan-t5-base'\n",
    "\n",
    "model = AutoModelForSeq2SeqLM.from_pretrained(model_name)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sPqQA3TT3l_I",
    "tags": []
   },
   "source": [
    "To perform encoding and decoding, you need to work with text in a tokenized form. **Tokenization** is the process of splitting texts into smaller units that can be processed by the LLM models. \n",
    "\n",
    "Download the tokenizer for the FLAN-T5 model using `AutoTokenizer.from_pretrained()` method. Parameter `use_fast` switches on fast tokenizer. At this stage, there is no need to go into the details of that, but you can find the tokenizer parameters in the [documentation](https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/auto#transformers.AutoTokenizer)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "sPqQA3TT3l_I",
    "tags": []
   },
   "outputs": [],
   "source": [
    "tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "Test the tokenizer encoding and decoding a simple sentence:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "sentence = \"What time is it, Tom?\"\n",
    "\n",
    "sentence_encoded = tokenizer(sentence, return_tensors='pt')\n",
    "\n",
    "sentence_decoded = tokenizer.decode(\n",
    "        sentence_encoded[\"input_ids\"][0], \n",
    "        skip_special_tokens=True\n",
    "    )\n",
    "\n",
    "print('ENCODED SENTENCE:')\n",
    "print(sentence_encoded[\"input_ids\"][0])\n",
    "print('\\nDECODED SENTENCE:')\n",
    "print(sentence_decoded)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now it's time to explore how well the base LLM summarizes a dialogue without any prompt engineering. **Prompt engineering** is an act of a human changing the **prompt** (input) to improve the response for a given task."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "for i, index in enumerate(example_indices):\n",
    "    dialogue = dataset['test'][index]['dialogue']\n",
    "    summary = dataset['test'][index]['summary']\n",
    "    \n",
    "    inputs = tokenizer(dialogue, return_tensors='pt')\n",
    "    output = tokenizer.decode(\n",
    "        model.generate(\n",
    "            inputs[\"input_ids\"], \n",
    "            max_new_tokens=50,\n",
    "        )[0], \n",
    "        skip_special_tokens=True\n",
    "    )\n",
    "    \n",
    "    print(dash_line)\n",
    "    print('Example ', i + 1)\n",
    "    print(dash_line)\n",
    "    print(f'INPUT PROMPT:\\n{dialogue}')\n",
    "    print(dash_line)\n",
    "    print(f'BASELINE HUMAN SUMMARY:\\n{summary}')\n",
    "    print(dash_line)\n",
    "    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\\n{output}\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can see that the guesses of the model make some sense, but it doesn't seem to be sure what task it is supposed to accomplish. Seems it just makes up the next sentence in the dialogue. Prompt engineering can help here."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a name='3'></a>\n",
    "## 3 - Summarize Dialogue with an Instruction Prompt\n",
    "\n",
    "Prompt engineering is an important concept in using foundation models for text generation. You can check out [this blog](https://www.amazon.science/blog/emnlp-prompt-engineering-is-the-new-feature-engineering) from Amazon Science for a quick introduction to prompt engineering."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a name='3.1'></a>\n",
    "### 3.1 - Zero Shot Inference with an Instruction Prompt\n",
    "\n",
    "In order to instruct the model to perform a task - summarize a dialogue - you can take the dialogue and convert it into an instruction prompt. This is often called **zero shot inference**.  You can check out [this blog from AWS](https://aws.amazon.com/blogs/machine-learning/zero-shot-prompting-for-the-flan-t5-foundation-model-in-amazon-sagemaker-jumpstart/) for a quick description of what zero shot learning is and why it is an important concept to the LLM model.\n",
    "\n",
    "Wrap the dialogue in a descriptive instruction and see how the generated text will change:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "for i, index in enumerate(example_indices):\n",
    "    dialogue = dataset['test'][index]['dialogue']\n",
    "    summary = dataset['test'][index]['summary']\n",
    "\n",
    "    prompt = f\"\"\"\n",
    "Summarize the following conversation.\n",
    "\n",
    "{dialogue}\n",
    "\n",
    "Summary:\n",
    "    \"\"\"\n",
    "\n",
    "    # Input constructed prompt instead of the dialogue.\n",
    "    inputs = tokenizer(prompt, return_tensors='pt')\n",
    "    output = tokenizer.decode(\n",
    "        model.generate(\n",
    "            inputs[\"input_ids\"], \n",
    "            max_new_tokens=50,\n",
    "        )[0], \n",
    "        skip_special_tokens=True\n",
    "    )\n",
    "    \n",
    "    print(dash_line)\n",
    "    print('Example ', i + 1)\n",
    "    print(dash_line)\n",
    "    print(f'INPUT PROMPT:\\n{prompt}')\n",
    "    print(dash_line)\n",
    "    print(f'BASELINE HUMAN SUMMARY:\\n{summary}')\n",
    "    print(dash_line)    \n",
    "    print(f'MODEL GENERATION - ZERO SHOT:\\n{output}\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is much better! But the model still does not pick up on the nuance of the conversations though."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise:**\n",
    "\n",
    "- Experiment with the `prompt` text and see how the inferences will be changed. Will the inferences change if you end the prompt with just empty string vs. `Summary: `?\n",
    "- Try to rephrase the beginning of the `prompt` text from `Summarize the following conversation.` to something different - and see how it will influence the generated output."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a name='3.2'></a>\n",
    "### 3.2 - Zero Shot Inference with the Prompt Template from FLAN-T5\n",
    "\n",
    "Let's use a slightly different prompt. FLAN-T5 has many prompt templates that are published for certain tasks [here](https://github.com/google-research/FLAN/tree/main/flan/v2). In the following code, you will use one of the [pre-built FLAN-T5 prompts](https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "for i, index in enumerate(example_indices):\n",
    "    dialogue = dataset['test'][index]['dialogue']\n",
    "    summary = dataset['test'][index]['summary']\n",
    "        \n",
    "    prompt = f\"\"\"\n",
    "Dialogue:\n",
    "\n",
    "{dialogue}\n",
    "\n",
    "What was going on?\n",
    "\"\"\"\n",
    "\n",
    "    inputs = tokenizer(prompt, return_tensors='pt')\n",
    "    output = tokenizer.decode(\n",
    "        model.generate(\n",
    "            inputs[\"input_ids\"], \n",
    "            max_new_tokens=50,\n",
    "        )[0], \n",
    "        skip_special_tokens=True\n",
    "    )\n",
    "\n",
    "    print(dash_line)\n",
    "    print('Example ', i + 1)\n",
    "    print(dash_line)\n",
    "    print(f'INPUT PROMPT:\\n{prompt}')\n",
    "    print(dash_line)\n",
    "    print(f'BASELINE HUMAN SUMMARY:\\n{summary}\\n')\n",
    "    print(dash_line)\n",
    "    print(f'MODEL GENERATION - ZERO SHOT:\\n{output}\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Notice that this prompt from FLAN-T5 did help a bit, but still struggles to pick up on the nuance of the conversation. This is what you will try to solve with the few shot inferencing."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a name='4'></a>\n",
    "## 4 - Summarize Dialogue with One Shot and Few Shot Inference\n",
    "\n",
    "**One shot and few shot inference** are the practices of providing an LLM with either one or more full examples of prompt-response pairs that match your task - before your actual prompt that you want completed. This is called \"in-context learning\" and puts your model into a state that understands your specific task.  You can read more about it in [this blog from HuggingFace](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "<a name='4.1'></a>\n",
    "### 4.1 - One Shot Inference\n",
    "\n",
    "Let's build a function that takes a list of `example_indices_full`, generates a prompt with full examples, then at the end appends the prompt which you want the model to complete (`example_index_to_summarize`).  You will use the same FLAN-T5 prompt template from section [3.2](#3.2). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def make_prompt(example_indices_full, example_index_to_summarize):\n",
    "    prompt = ''\n",
    "    for index in example_indices_full:\n",
    "        dialogue = dataset['test'][index]['dialogue']\n",
    "        summary = dataset['test'][index]['summary']\n",
    "        \n",
    "        # The stop sequence '{summary}\\n\\n\\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.\n",
    "        prompt += f\"\"\"\n",
    "Dialogue:\n",
    "\n",
    "{dialogue}\n",
    "\n",
    "What was going on?\n",
    "{summary}\n",
    "\n",
    "\n",
    "\"\"\"\n",
    "    \n",
    "    dialogue = dataset['test'][example_index_to_summarize]['dialogue']\n",
    "    \n",
    "    prompt += f\"\"\"\n",
    "Dialogue:\n",
    "\n",
    "{dialogue}\n",
    "\n",
    "What was going on?\n",
    "\"\"\"\n",
    "        \n",
    "    return prompt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "Construct the prompt to perform one shot inference:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "example_indices_full = [40]\n",
    "example_index_to_summarize = 200\n",
    "\n",
    "one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)\n",
    "\n",
    "print(one_shot_prompt)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "Now pass this prompt to perform the one shot inference:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "summary = dataset['test'][example_index_to_summarize]['summary']\n",
    "\n",
    "inputs = tokenizer(one_shot_prompt, return_tensors='pt')\n",
    "output = tokenizer.decode(\n",
    "    model.generate(\n",
    "        inputs[\"input_ids\"],\n",
    "        max_new_tokens=50,\n",
    "    )[0], \n",
    "    skip_special_tokens=True\n",
    ")\n",
    "\n",
    "print(dash_line)\n",
    "print(f'BASELINE HUMAN SUMMARY:\\n{summary}\\n')\n",
    "print(dash_line)\n",
    "print(f'MODEL GENERATION - ONE SHOT:\\n{output}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "<a name='4.2'></a>\n",
    "### 4.2 - Few Shot Inference\n",
    "\n",
    "Let's explore few shot inference by adding two more full dialogue-summary pairs to your prompt."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "example_indices_full = [40, 80, 120]\n",
    "example_index_to_summarize = 200\n",
    "\n",
    "few_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)\n",
    "\n",
    "print(few_shot_prompt)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "Now pass this prompt to perform a few shot inference:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "summary = dataset['test'][example_index_to_summarize]['summary']\n",
    "\n",
    "inputs = tokenizer(few_shot_prompt, return_tensors='pt')\n",
    "output = tokenizer.decode(\n",
    "    model.generate(\n",
    "        inputs[\"input_ids\"],\n",
    "        max_new_tokens=50,\n",
    "    )[0], \n",
    "    skip_special_tokens=True\n",
    ")\n",
    "\n",
    "print(dash_line)\n",
    "print(f'BASELINE HUMAN SUMMARY:\\n{summary}\\n')\n",
    "print(dash_line)\n",
    "print(f'MODEL GENERATION - FEW SHOT:\\n{output}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "In this case, few shot did not provide much of an improvement over one shot inference.  And, anything above 5 or 6 shot will typically not help much, either.  Also, you need to make sure that you do not exceed the model's input-context length which, in our case, if 512 tokens.  Anything above the context length will be ignored.\n",
    "\n",
    "However, you can see that feeding in at least one full example (one shot) provides the model with more information and qualitatively improves the summary overall."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "**Exercise:**\n",
    "\n",
    "Experiment with the few shot inferencing.\n",
    "- Choose different dialogues - change the indices in the `example_indices_full` list and `example_index_to_summarize` value.\n",
    "- Change the number of shots. Be sure to stay within the model's 512 context length, however.\n",
    "\n",
    "How well does few shot inferencing work with other examples?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "<a name='5'></a>\n",
    "## 5 - Generative Configuration Parameters for Inference"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "You can change the configuration parameters of the `generate()` method to see a different output from the LLM. So far the only parameter that you have been setting was `max_new_tokens=50`, which defines the maximum number of tokens to generate. A full list of available parameters can be found in the [Hugging Face Generation documentation](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig). \n",
    "\n",
    "A convenient way of organizing the configuration parameters is to use `GenerationConfig` class. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "**Exercise:**\n",
    "\n",
    "Change the configuration parameters to investigate their influence on the output. \n",
    "\n",
    "Putting the parameter `do_sample = True`, you activate various decoding strategies which influence the next token from the probability distribution over the entire vocabulary. You can then adjust the outputs changing `temperature` and other parameters (such as `top_k` and `top_p`). \n",
    "\n",
    "Uncomment the lines in the cell below and rerun the code. Try to analyze the results. You can read some comments below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "generation_config = GenerationConfig(max_new_tokens=50)\n",
    "# generation_config = GenerationConfig(max_new_tokens=10)\n",
    "# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)\n",
    "# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)\n",
    "# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=1.0)\n",
    "\n",
    "inputs = tokenizer(few_shot_prompt, return_tensors='pt')\n",
    "output = tokenizer.decode(\n",
    "    model.generate(\n",
    "        inputs[\"input_ids\"],\n",
    "        generation_config=generation_config,\n",
    "    )[0], \n",
    "    skip_special_tokens=True\n",
    ")\n",
    "\n",
    "print(dash_line)\n",
    "print(f'MODEL GENERATION - FEW SHOT:\\n{output}')\n",
    "print(dash_line)\n",
    "print(f'BASELINE HUMAN SUMMARY:\\n{summary}\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Comments related to the choice of the parameters in the code cell above:\n",
    "- Choosing `max_new_tokens=10` will make the output text too short, so the dialogue summary will be cut.\n",
    "- Putting `do_sample = True` and changing the temperature value you get more flexibility in the output."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As you can see, prompt engineering can take you a long way for this use case, but there are some limitations. Next, you will start to explore how you can use fine-tuning to help your LLM to understand a particular use case in better depth!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "availableInstances": [
   {
    "_defaultOrder": 0,
    "_isFastLaunch": true,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 4,
    "name": "ml.t3.medium",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 1,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.t3.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 2,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.t3.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 3,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.t3.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 4,
    "_isFastLaunch": true,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.m5.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 5,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.m5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 6,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.m5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 7,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.m5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 8,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.m5.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 9,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.m5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 10,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.m5.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 11,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.m5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 12,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.m5d.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 13,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.m5d.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 14,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.m5d.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 15,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.m5d.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 16,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.m5d.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 17,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.m5d.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 18,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.m5d.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 19,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.m5d.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 20,
    "_isFastLaunch": false,
    "category": "General purpose",
    "gpuNum": 0,
    "hideHardwareSpecs": true,
    "memoryGiB": 0,
    "name": "ml.geospatial.interactive",
    "supportedImageNames": [
     "sagemaker-geospatial-v1-0"
    ],
    "vcpuNum": 0
   },
   {
    "_defaultOrder": 21,
    "_isFastLaunch": true,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 4,
    "name": "ml.c5.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 22,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 8,
    "name": "ml.c5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 23,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.c5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 24,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.c5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 25,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 72,
    "name": "ml.c5.9xlarge",
    "vcpuNum": 36
   },
   {
    "_defaultOrder": 26,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 96,
    "name": "ml.c5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 27,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 144,
    "name": "ml.c5.18xlarge",
    "vcpuNum": 72
   },
   {
    "_defaultOrder": 28,
    "_isFastLaunch": false,
    "category": "Compute optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.c5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 29,
    "_isFastLaunch": true,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.g4dn.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 30,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.g4dn.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 31,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.g4dn.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 32,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.g4dn.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 33,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.g4dn.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 34,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.g4dn.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 35,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 61,
    "name": "ml.p3.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 36,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 244,
    "name": "ml.p3.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 37,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 488,
    "name": "ml.p3.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 38,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 768,
    "name": "ml.p3dn.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 39,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.r5.large",
    "vcpuNum": 2
   },
   {
    "_defaultOrder": 40,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.r5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 41,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.r5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 42,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.r5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 43,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.r5.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 44,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.r5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 45,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 512,
    "name": "ml.r5.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 46,
    "_isFastLaunch": false,
    "category": "Memory Optimized",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 768,
    "name": "ml.r5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 47,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 16,
    "name": "ml.g5.xlarge",
    "vcpuNum": 4
   },
   {
    "_defaultOrder": 48,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.g5.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 49,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 64,
    "name": "ml.g5.4xlarge",
    "vcpuNum": 16
   },
   {
    "_defaultOrder": 50,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 128,
    "name": "ml.g5.8xlarge",
    "vcpuNum": 32
   },
   {
    "_defaultOrder": 51,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 1,
    "hideHardwareSpecs": false,
    "memoryGiB": 256,
    "name": "ml.g5.16xlarge",
    "vcpuNum": 64
   },
   {
    "_defaultOrder": 52,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 192,
    "name": "ml.g5.12xlarge",
    "vcpuNum": 48
   },
   {
    "_defaultOrder": 53,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 4,
    "hideHardwareSpecs": false,
    "memoryGiB": 384,
    "name": "ml.g5.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 54,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 768,
    "name": "ml.g5.48xlarge",
    "vcpuNum": 192
   },
   {
    "_defaultOrder": 55,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 1152,
    "name": "ml.p4d.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 56,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 8,
    "hideHardwareSpecs": false,
    "memoryGiB": 1152,
    "name": "ml.p4de.24xlarge",
    "vcpuNum": 96
   },
   {
    "_defaultOrder": 57,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 32,
    "name": "ml.trn1.2xlarge",
    "vcpuNum": 8
   },
   {
    "_defaultOrder": 58,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 512,
    "name": "ml.trn1.32xlarge",
    "vcpuNum": 128
   },
   {
    "_defaultOrder": 59,
    "_isFastLaunch": false,
    "category": "Accelerated computing",
    "gpuNum": 0,
    "hideHardwareSpecs": false,
    "memoryGiB": 512,
    "name": "ml.trn1n.32xlarge",
    "vcpuNum": 128
   }
  ],
  "instance_type": "ml.m5.2xlarge",
  "kernelspec": {
   "display_name": "Python 3 (Data Science 3.0)",
   "language": "python",
   "name": "python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:081325390199:image/sagemaker-data-science-310-v1"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
