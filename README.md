# pruning-defense
Pruning defense for backdoor attacks

The pruned models are stored in the [models](https://github.com/ajn313/pruning-defense/tree/main/models) folder.
They are titled gn-x2.h5, gn-x4.h5, and gn-x10.h5 based on the percentage accuracy loss allowed, i.e. X=(2, 4, 10). 
To evaluate, the same format from the CSAW repo is used.
For example:
python3 eval.py data/cl/valid.h5 data/bd/bd_valid.h5 models/gn-x10.h5

Data was obtained from [here](https://drive.google.com/drive/folders/1Rs68uH8Xqa4j6UxG53wzD0uyI8347dSq)
