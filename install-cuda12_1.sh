conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d
cd submodules
git clone https://github.com/ashawkey/diff-gaussian-rasterization --recursive
cd ..
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17 