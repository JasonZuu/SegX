
python grid_search.py --dataset ham --task cls --model resnet --pretrained
python grid_search.py --dataset ham --task cls --model densenet --pretrained
python grid_search.py --dataset chestx --task cls --model resnet --pretrained
python grid_search.py --dataset chestx --task cls --model densenet --pretrained

python grid_search.py --dataset ham --task seg --model unet --pretrained
python grid_search.py --dataset chestx --task seg --model unet --pretrained