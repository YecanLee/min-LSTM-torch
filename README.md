## min-LSTM-torch
officially unofficial PyTorch Implementation of "Were RNNs All We Needed?"

This code includes the pytorch implementation of the paper: [Were RNNs All We Needed?](https://arxiv.org/pdf/2410.01201)

## Installation

```
pip install -r requirements.txt
```

## Training

```
python train.py
```

## The result will look like this:


```
668/668 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9326 - loss: 0.1749 - val_acc: 0.7870 - val_loss: 0.6409
Epoch 10/100
668/668 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9635 - loss: 0.1013 - val_acc: 0.7887 - val_loss: 0.8138
Epoch 11/100
668/668 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - acc: 0.9813 - loss: 0.0526 - val_acc: 0.7756 - val_loss: 1.0768

```
