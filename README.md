# SIPN

PyTorch implementation of “Graph-based Deep Learning with Phase Normalization for Hand Gesture Recognition”.

# Data Preparation

- SHREC
  - Download the SHREC data from http://www-rech.telecom-lille.fr/shrec2017-hand/
  - Generate the train/test splits with `python prepare/shrec/gendata.py`
- DHG
  - Download the DHG data from the http://www-rech.telecom-lille.fr/DHGdataset/
  - Generate the train/test splits with `python prepare/dhg/gendata.py`

## Training :

### Shrec

```
python main.py --config configs/shrec17/14/j.yaml 
```

### DHG

```
python main.py --config  configs/dgh/dgh14/joint.yaml
```

## Testing

```
python main.py --config configs/nut60/xsub/joint.yaml --phase test --weights xxxx
```

## Ensemble

```
python ensemble.py --config ensemble.yaml
```

The baseline for our project:  [DSTSA-GCN](https://github.com/HuCui2022/DSTSA-GCN_Gesture)