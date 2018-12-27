Demo-A3C-Doom

### captures
- A2C using Gluon Model  
![](https://github.com/FutureGameCritic/Demo-A3C-Doom/blob/master/capture/a2c_on_gluon.png)  

## install
```
conda create -y -n dad python=3.6
source activate dad
sudo pip3 install -r requirements.txt
```
`@Warnning`
conda is recommanded to install vizdoom. but, you can use naive-pip3 by installing all os-dependency that following guide in `os/~` folder.
```
virtualenv -p python3.6 dad
. dad/bin/activate
sudo pip3 install -r requirements.txt
```

## train
```
python3 train.py
```

## evaluation
```
python3 eval.py
```

## test
if you want to play doom by random actions, just run `test.py` script


## models
- common
    - basic block : conv1 -> conv2 -> dense(256)
    - input : stack 4 frames * (84 * 84) grayscaled label data
- A2C
    - 
- A3C
    - multi-threading
    
## usage
- `gluon`
- `tensorflow`
- `visdoom`