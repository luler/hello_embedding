# 基于BAAI/bge-large-zh-v1.5模型的文本向量提取工具

参考模型地址：https://huggingface.co/BAAI/bge-large-zh-v1.5

## 安装

### 方式一：本地python环境启动

安装所需软件包

``` 
pip install -i https://mirror.baidu.com/pypi/simple -r requirements.txt
```

启动

``` 
python main.py
```

### 方式二：docker-compose一键安装

```
docker-compose up -d
```

注意：初次启动时需要下载模型文件，需要等待一段时间，具体看网速，可查看运行过程相关输出日志

## 相关接口

提供的接口：http://127.0.0.1:8000/api/embeddings

与openai的embeddings接口兼容

参考：https://platform.openai.com/docs/guides/embeddings

![](example.png)