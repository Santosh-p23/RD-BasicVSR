conda create --name major_proj python=3.9
conda activate major_proj
pip install numpy mmcv mmedit pillow
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia