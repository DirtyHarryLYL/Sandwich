# Semantic Alignment

To construct a shared semantic space for various datasets, semantic consistency should be maintained. The class definitions of datasets are various, but they can be mapped to our semantic space with the fewest semantic damages. 

You can use semantic alignment not only for collected activities but also for any new activities you are interested in :)

<p align='center'>
    <img src="../img/semantic_alignment.png", height="300">
</p>

## Query Our Collected Activities
We provide the annotated results for batch1/2/3/4(...more coming) data.
You can query our collected activities via [query_collected.ipynb](./semantic_alignment/query_collected.ipynb).

## Query New Activities
### Utilize Collected Knowledge
With the collected activities and their mapping, you can query new activities via their semantic correlation with the collected ones.

The script is given in [query_new.ipynb](./semantic_alignment/query_new.ipynb).

### Utilize LLM Knowledge
With the recent progress of LLM, automatic semantic alignment can be established via LLM querying. We provide the script in [align_with_clip+gpt.py](./semantic_alignment/align_with_clip+gpt.py) and some results for batch4 data in [dataset_labels_align_result](./semantic_alignment/dataset_labels_align_result).

