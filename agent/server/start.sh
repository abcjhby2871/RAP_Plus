cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=4
source ~/anaconda3/bin/activate rap
python app.py 
