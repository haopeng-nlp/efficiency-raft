TASKS="ade_corpus_v2 banking_77 neurips_impact_statement_risks one_stop_english overruling systematic_review_inclusion tai_safety_research terms_of_service tweet_eval_hate twitter_complaints"

for TASK in $TASKS; do
    # AdaBoostClassifier
    efficiency-benchmark submit --name $RAFT-$TASK-adaboost --task raft::$TASK  --max_batch_size 1 -- python entrypoint.py with classifier_name=AdaBoostClassifier

    # distilgpt2
    efficiency-benchmark submit --name $RAFT-$TASK-distilgpt2 --task raft::$TASK  --max_batch_size 1 -- python entrypoint.py with classifier_name=TransformersCausalLMClassifier task=$TASK classifier_kwargs='{"model_type":"distilgpt2", "num_prompt_training_examples": 25, "use_task_specific_instructions": True, "do_semantic_selection": True, "add_prefixes": False}' 

    # gpt2
    efficiency-benchmark submit --name $RAFT-$TASK-gpt2 --task raft::$TASK  --max_batch_size 1 -- python entrypoint.py with classifier_name=TransformersCausalLMClassifier task=$TASK classifier_kwargs='{"model_type":"gpt2", "num_prompt_training_examples": 25, "use_task_specific_instructions": True, "do_semantic_selection": True, "add_prefixes": False}'

    # zero shot bart-large-mnli
    efficiency-benchmark submit --name $RAFT-$TASK-bart --task raft::$TASK  --max_batch_size 1 -- python entrypoint.py with classifier_name=TransformersZeroShotPipelineClassifier task=$TASK classifier_kwargs='{"model_type":"facebook/bart-large-mnli"}'

done
