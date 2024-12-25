# GPT for Text Generation

Implemented a decoder-only Transformer model on the SCAN dataset using PyTorch. The goal of SCAN is to translate commands presented in simplified natural language into a sequence of actions. In this translation task, the generation model will take a command sentence as input and output the corresponding action sequence. For more details, please refer to the dataset available [here](https://huggingface.co/datasets/scan). You can also refer to its [original paper](https://arxiv.org/pdf/1711.00350.pdf) for the introduction of the task.

Running the starting code will directly download the SCAN dataset automatically. Other splits can be used by simply setting the CLI argument ”data split” to the names of the splits.

Used the data split ‘length’ here. The split is designed to evaluate the model's ability to generalize and produce accurate outputs for longer action sequences that contain combinations of familiar elements (verbs, modifiers, and conjunctions) seen during training but in novel combinations or lengths. This evaluation assesses the model's capacity to effectively combine and extend learned actions and linguistic structures to generate coherent and contextually appropriate sequences, even when faced with previously unseen combinations or lengths of actions.
