pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c iopath iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

cd submodules
git clone https://github.com/ashawkey/diff-gaussian-rasterization --recursive
cd ..
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch