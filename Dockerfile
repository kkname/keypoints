# 使用官方的NVIDIA CUDA基础镜像，确保GPU环境正确
# 您可以根据您服务器的CUDA版本选择，11.8.0是一个稳定且现代的选择
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 设置环境变量，避免安装过程中的交互式提问
ENV DEBIAN_FRONTEND=noninteractive

# 安装基础系统依赖，包括git和一些open3d可能需要的库
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 创建工作目录
WORKDIR /VoxelKP

# 复制环境配置文件到镜像中
COPY environment.yml .

# 使用conda根据环境文件创建并激活Python环境
# 这一步会自动下载并安装所有您本地的库，包括PyTorch, Open3D等
RUN conda env create -f environment.yml

# 设置SHELL，以便后续命令能在conda环境中执行
SHELL ["conda", "run", "-n", "py3.8", "/bin/bash", "-c"]

# 将您的项目代码（除了.dockerignore中忽略的）复制到工作目录
COPY . .

# 编译pcdet的自定义CUDA操作
# 这一步至关重要，它确保了spconv等库在Docker环境中被正确编译
RUN python setup.py develop

# 设置容器启动后默认进入的目录
WORKDIR /VoxelKP

# 设置容器启动后的默认命令（进入bash交互环境）
CMD ["/bin/bash"]