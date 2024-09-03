# Sandwich
## Bidirectional Mapping between Action Physical-Semantic Space
Maintained by Yong-Lu Li, Xiaoqian Wu, Xinpeng Liu, Yiming Dou, and Cewu Lu.
- **Pangea** (CVPR 2024 Highlight): Unified action semantic space (label system) tp aggregate all action datasets into ONE. [Pangea](https://github.com/DirtyHarryLYL/Sandwich/tree/main/pangea_data)

<p align='center'>
    <img src="https://github.com/DirtyHarryLYL/Sandwich/blob/main/asset/sandwich.png", height="400">
</p>

#### **News**: (2023.2.27) Pangea will appear at CVPR 2024!

Sandwich is a project for general human activity understanding. It consists of:
- A novel structured semantic space to re-explain all the action classes from existing action datasets of multiple modalities.
- A large-scale multi-modal action database, **Pangea**, aggregating a plenty of existing action datasets.
- A bidirectional mapping system to learn the bi-mapping between human action physical-semantic space and facilitate the action recognition and generation.

## Pangea Data
#### We are gathering more and more human action datasets and linking their label systems to Pangea (CVPR 2024), for more details please refer to this [page](https://github.com/DirtyHarryLYL/Sandwich/tree/main/pangea_data). 
- Given our new hierarchical semantic system, you can train big models on all action datasets under ONE label system with principled knowledge! 

## Installation
For python environment, see [requirements.txt](requirements.txt).

The data is given [here](https://drive.google.com/drive/folders/1VO9M79JAOtf9rTqYHx_HWphOvAvqNGFa?usp=sharing). 
Download the folder, rename it as "Data", and put it in the root dir.

## P2S mapping
See [p2s_mapping](p2s_mapping). Run [eval_node.sh](p2s_mapping/eval_node.sh) to infer verb node labels.

## Pose2SMPL used in our 3D human data curation.
[Pose2SMPL](https://github.com/Dou-Yiming/Pose_to_SMPL)
