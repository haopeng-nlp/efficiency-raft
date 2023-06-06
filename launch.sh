tasks="ade_corpus_v2 banking_77 neurips_impact_statement_risks one_stop_english overruling systematic_review_inclusion tai_safety_research terms_of_service tweet_eval_hate twitter_complaints"

for TASK in $tasks; do
    echo bart-large-mnli-$TASK
    efficiency-benchmark submit --task raft::$TASK  --max_batch_size 1  -- python entrypoint.py with classifier_name=TransformersZeroShotPipelineClassifier task=$TASK classifier_kwargs='{"model_type":"facebook/bart-large-mnli", "config": "$TASK"}'
done
