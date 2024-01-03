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

## Aligned Dataset Statistic
Currently, we have aligned **10642** action labels over **52** datasets.

| Dataset             | Action Classes  | Collected | Aligned  |
|---------------------|-----------------|-----------|----------|
| Willow Action       | 7               | Y         | Y        |
| Phrasal Recognition | 10              | Y         | Y        |
| Stanford 40 Actions | 40              | Y         | Y        |
| MPII                | 410             | Y         | Y        |
| HICO                | 600             | Y         | Y        |
| HAKE                | 156             | Y         | Y        |
| HMDB51              | 51              | Y         | Y        |
| HAA500              | 500             | Y         | Y        |
| AVA                 | 80              | Y         | Y        |
| YouTube Action      | 11              | Y         | Y        |
| ASLAN               | 432             | Y         | Y        |
| UCF101              | 101             | Y         | Y        |
| Olympic Sports      | 16              | Y         | Y        |
| Penn Action         | 15              | Y         | Y        |
| Charades            | 157             | Y         | Y        |
| Charades-Ego        | 157             | Y         | Y        |
| ActivityNet         | 200             | Y         | Y        |
| HACS                | 200             | Y         | Y        |
| Home Action Genome  | 453             | Y         | Y        |
| Kinetics      | 700            | Y         | Y        |
| MOD20         | 20             | Y         | Y        |
| IKEA_ASM      | 33             | Y         | Y        |
| FineAction    | 106            | Y         | Y        |
| CAP           | 459            | Y         | Y        |
| CrossTask     | 704            | Y         | Y        |
| FineGym(530)  | 530            | Y         | Y        |
| FineGym(99)   | 99             | Y         | Y        |
| UCF_Crime     | 13             | Y         | Y        |
| XD-Violence   | 6              | Y         | Y        |
| FineGym(288)  | 288            | Y         | Y        |
| YouTube8M     | 493            | Y         | Y        |
| Sports-1M     | 487            | Y         | Y        |
| THUMOS        | 101            | Y         | Y        |
| Action_Genome | 157            | Y         | Y        |
| MultiTHUMOS   | 65             | Y         | Y        |
| HVU           | 739            | Y         | Y        |
| RareAct       | 136            | Y         | Y        |
| COIN          | 778            | Y         | Y        |
| Hollywood2    | 12             | Y         | Y        |
| MovieNet      | 80             | Y         | Y        |
| MultiSports   | 66             | Y         | Y        |
| InHARD        | 14             | Y         | TODO     |
| Jester        | 27             | Y         | TODO     |
| MECCANO       | 61             | Y         | TODO     |
| MPII_Cooking  | 65             | Y         | TODO     |
| UAV_Human     | 155            | Y         | TODO     |
| WLASL         | 100            | Y         | TODO     |
| Breakfast     | 49             | Y         | TODO     |
| Ego4d         | 77             | Y         | TODO     |
| EgteaGaze     | 106            | Y         | TODO     |
| Epic_Kitchen  | 97             | Y         | TODO     |
| FiftySalads   | 17             | Y         | TODO     |
| IKEA          | 32             | Y         | TODO     |
| SomethingElse | 174            | Y         | TODO     |
