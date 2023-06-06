## install

`pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html`

then `pip install -e .`

## usage

```bash
# remove debug=True for efficiency benchmark


# AdaBoostClassifier
python entrypoint.py with classifier_name=AdaBoostClassifier debug=True


# distilgpt2
python entrypoint.py with classifier_name=TransformersCausalLMClassifier task=ade_corpus_v2 classifier_kwargs='{"model_type":"distilgpt2", "num_prompt_training_examples": 25, "use_task_specific_instructions": True, "do_semantic_selection": True, "config": "ade_corpus_v2", "add_prefixes": False}' debug=True

# gpt2
python entrypoint.py with classifier_name=TransformersCausalLMClassifier task=ade_corpus_v2 classifier_kwargs='{"model_type":"gpt2", "num_prompt_training_examples": 25, "use_task_specific_instructions": True, "do_semantic_selection": True, "config": "ade_corpus_v2", "add_prefixes": False}' debug=True

# zero shot bart-large-mnli
python entrypoint.py with classifier_name=TransformersZeroShotPipelineClassifier task=ade_corpus_v2 classifier_kwargs='{"model_type":"facebook/bart-large-mnli", "config": "ade_corpus_v2"}' debug=True

```

## efficiency benchmark

```bash


efficiency-benchmark run --task raft::ade_corpus_v2  --max_batch_size 1 --scenario accuracy  -- xxx (add above command)

# e.g. 
efficiency-benchmark run --task raft::ade_corpus_v2  --max_batch_size 1 --scenario accuracy  -- python entrypoint.py with classifier_name=TransformersZeroShotPipelineClassifier task=ade_corpus_v2 classifier_kwargs='{"model_type":"facebook/bart-large-mnli", "config": "ade_corpus_v2"}' 
```