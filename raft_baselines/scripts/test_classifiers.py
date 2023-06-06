import datasets

from raft_baselines.classifiers import NaiveBayesClassifier, AdaBoostClassifier, TransformersCausalLMClassifier, TransformersZeroShotPipelineClassifier

train = datasets.load_dataset(
    "ought/raft", "ade_corpus_v2", split="train"
)

classifier = NaiveBayesClassifier(train)

print(classifier.classify({'Sentence': 'No regional side effects were noted.', 'ID': 0, 'Label': 2}))
# print(classifier.classify({"Paper title": "CNN research", "Impact statement": "test2"}))

classifier = AdaBoostClassifier(train)

print(classifier.classify({'Sentence': 'No regional side effects were noted.', 'ID': 0, 'Label': 2}))
# print(classifier.classify({"Paper title": "CNN research", "Impact statement": "test2"}))
# classifier = SVMClassifier(train)

# print(classifier.classify({'Sentence': 'No regional side effects were noted.', 'ID': 0, 'Label': 2}))
# # print(classifier.classify({"Paper title": "CNN research", "Impact statement": "test2"}))


TASK = "ade_corpus_v2"
raft_dataset = datasets.load_dataset("ought/raft", name=TASK)

test_dataset = raft_dataset["test"]
batch_examples = test_dataset[:5]
print(batch_examples)

batch = []
    
# for be in batch_examples:
#     print(be)
#     print(classifier.classify(be))
#     print()
first_test_example = test_dataset[0]
print(first_test_example)
# delete the 0 Label
del first_test_example["Label"]

# for model in ["distilgpt2", "gpt2", 'EleutherAI/gpt-neo-2.7B']:
for model in ["distilgpt2", "gpt2", 'EleutherAI/gpt-neo-125m']:
    classifier = TransformersCausalLMClassifier(
        model_type=model,             # The model to use from the HF hub
        training_data=raft_dataset["train"],            # The training data
        num_prompt_training_examples=25,     # See raft_predict.py for the number of training examples used on a per-dataset basis in the GPT-3 baselines run.
                                            # Note that it may be better to use fewer training examples and/or shorter instructions with other models with smaller context windows.
        add_prefixes=(TASK=="banking_77"),   # Set to True when using banking_77 since multiple classes start with the same token
        config=TASK,                         # For task-specific instructions and field ordering
        use_task_specific_instructions=True,
        do_semantic_selection=True,
    )

    # probabilities for all classes
    # output_probs = classifier.classify(first_test_example, should_print_prompt=True)
    output_probs = classifier.classify(first_test_example)
    print(model, output_probs)

for model in ["facebook/bart-large-mnli", 'cross-encoder/nli-deberta-base']:
    classifier = TransformersZeroShotPipelineClassifier(
        training_data=raft_dataset["train"],
        config=TASK,
        model="facebook/bart-large-mnli",
    )
    output_probs = classifier.classify(first_test_example)
    print(model, output_probs)

