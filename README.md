# sklearn-predict
机器学习数据，预测趋势并画图,里面有多种预测算法，需要有机器学习基础的人才能使用。

## 数据
* 按powerdata.csv的格式组织,程序调用的是"powerdata.csv"文件,文件中数据每列为间隔15分钟的值，一天24小时，一共24乘4列。如果时间格式不是这样的，可以在svm-prediction中开头修改“96”这个值做调整

## 环境
* ubuntu 18.04
* python 3.7.*

## 安装
* `sudo chmod u+x install.sh`
* `sudo bash ./install.sh`
* `sudo vim ~/.bash_profile` 
```
   export PATH="$HOME/.pyenv/bin:$PATH"  
   eval "$(pyenv init -)" 
   eval "$(pyenv virtualenv-init -)"  
```
* `source ~/.bash_profile`

## python3.7.4 version virtualenv && install python libraries
```pyenv install 3.7.4
   pyenv virtualenv 3.7.4 svm-prediction-venv
   pyenv activate svm-prediction-venv
   pipenv install
```

## 调用
* `python predict.py --help`
```
Usage: predict.py [OPTIONS]

Options:
  --train INTEGER  train size(day).
  --test INTEGER   test size(day).
  --filepath TEXT  file path
  --help           Show this message and exit.
```

## 效果图
* SVR chart   
![SVR chart](https://github.com/zhengze/svm-prediction/blob/develop/images/svr.png)  
* GradientBoostingRegressor chart   
![GradientBoostingRegressor chart](https://github.com/zhengze/svm-prediction/blob/develop/images/jueceshu.png)
