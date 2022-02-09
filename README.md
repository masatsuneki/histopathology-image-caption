# histopathology image caption

A dataset of 262,777 patches extracted from 991 H&E-stained gastric slides with Adenocarcinoma subtypes paired with captions extracted from medical reports. For more details see [paper](https://openreview.net/forum?id=9gKn7SDb83v).


[captions.csv](captions.csv) contains `id,subtype,text` columns, where `id` designates the whole slide image id from which the patches were extracted. The patches filenames have `id` in the prefix as follows:  `{id}_{random hash}.jpg`. The patches can be downloaded from [here](https://zenodo.org/record/6550925).


![](captions_adc.jpg)

Dataset is provided for research use only.

If you use this Dataset, please cite:

```
@misc{tsuneki2022inference,
      title={Inference of captions from histopathological patches}, 
      author={Masayuki Tsuneki and Fahdi Kanavati},
      year={2022},
      eprint={2202.03432},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Running training script for baseline model

build the docker image

```
docker build -t histo-captions .
```

Assuming the the patches have been extracted at `/mnt/data/patches/x20` and the `captions.csv` file is at `/mnt/data/captions.csv`, you can run it with default settings with

```
docker run -v /mnt/data:/data -it histo-captions  python train.py 
```

To check for available options, run

```
docker run -v /mnt/data:/data -it histo-captions  python train.py --help
```