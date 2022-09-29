# Faster_RCNN_pytorch

> [Paper_Review](https://inhopp.github.io/paper/Paper10/)

| original_image | Detection |
|:-:|:-:|
| ![orange_91](https://user-images.githubusercontent.com/96368476/192968601-6dae1a8b-e8d6-4df0-a815-9a48d488885e.jpg) | ![detection](https://user-images.githubusercontent.com/96368476/192968604-d6a4fdb6-44a8-4de0-a515-424bc0c29eb4.png) |



<br>

## Repository Directory 

``` python 
├── Faster_RCNN_pytorch
        ├── datasets
        │     └── fruits
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── model
        │     ├── utils.py
        │     │      ├── bbox_tool.py
        │     │      └── creat_tool.py      
        │     ├── region_proposal_network.py          
        │     └── faster_rcnn.py        
        ├── option.py
        ├── utils.py
        ├── calc_mAP.py
        ├── train.py
        ├── trainer.py
        └── inference.py
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get annotations
- `model/utils` : utils for models (about bounding box and ROI)
- `model/region_proposal_network.py` : Define RPN 
- `model/faster_rcnn.py` : Define backbone(VGG16), head(Fast RCNN) and Faster RCNN
- `option.py` : Environment setting
- `utils` : general utils
- `calc_mAP` : Calculation AP and mAP
- `trainer.py` : Define loss and model forward (because multi-module networks)


<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
https://github.com/inhopp/Faster_RCNN_pytorch.git
pip3 install requirements.txt
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --data_name {}(default: fruits) \
    --lr {}(default: 0.0005) \
    --n_epoch {}(default: 10) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 1) \ # support only batch_size=1
    --eval_batch_size {}(default: 1) \ # support only batch_size=1
```

### testset inference
```python
python3 inference.py
    --device {}(defautl: cpu) \
    --data_name {}(default: rsp_data) \
    --num_workers {}(default: 4) \
    --eval_batch_size {}(default: 1) \ # support only batch_size=1
```

<br>

#### Main Reference
https://github.com/shkim5616/faster-rcnn-for-studying
