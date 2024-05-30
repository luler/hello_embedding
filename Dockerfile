# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt requirements.txt

# 安装依赖
RUN pip install -i https://mirror.baidu.com/pypi/simple -r requirements.txt

# 复制当前目录中的文件到工作目录中
COPY . .

#默认模型
ENV EMBEDDING_PATH=BAAI/bge-large-zh-v1.5

# 下载模型
RUN python -c "from sentence_transformers import SentenceTransformer; import torch; import os; SentenceTransformer(os.environ.get('EMBEDDING_PATH'), device=('cuda' if torch.cuda.is_available() else 'cpu'))"

# 暴露端口
EXPOSE 8000

# 设置启动命令
CMD ["python", "main.py"]