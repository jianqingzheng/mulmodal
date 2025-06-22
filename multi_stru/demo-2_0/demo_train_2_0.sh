source  /home/jachin/anaconda3/bin/activate multi_stru
# export CUDA_VISIBLE_DEVICES=1,3
cd "$(dirname "$0")"

nohup python -u /home/jachin/anaconda3/envs/multi_stru/bin/python 2_5.training-stru_shanghai.py > demo20_out.log &     # small data with only sequence input
# nohup python -u /home/jachin/anaconda3/envs/multi_stru/bin/python 2_5_1.training-stru_shanghai.py > demo20_out.log &     # small data with sequence and structure input
# nohup python -u /home/jachin/anaconda3/envs/multi_stru/bin/python 2_5_2.training-large-stru_shanghai.py > demo20_out.log &     # large data with only sequence and structure input