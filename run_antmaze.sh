export D4RL_SUPPRESS_IMPORT_ERROR=1

NUM_GPUS=4
ENV_LIST=(
	"antmaze-umaze-v2"
	# "antmaze-umaze-diverse-v2"
	# "antmaze-medium-play-v2"
	# "antmaze-medium-diverse-v2"
	# "antmaze-large-play-v2"
	# "antmaze-large-diverse-v2"

ALPHA_LIST=(
	0.1
	# 1.0
	# 2.0
)

EPSILON_LIST=(
	-1.0
	# -2.0
	# -3.0
)


let "gpu=0"
for env in ${ENV_LIST[@]}; do
for alpha in ${ALPHA_LIST[@]}; do
for epsilon in ${EPSILON_LIST[@]}; do
XLA_PYTHON_CLIENT_MEM_FRACTION=.10 CUDA_VISIBLE_DEVICES=$gpu python train_offline.py \
--env_name $env \
--config=configs/antmaze_config.py \
--alg "PORelDICE" \
--alpha $alpha \
--epsilon $epsilon \
--max_steps 1000000 \
--log_interval 100 \
--eval_interval 100000 \
--eval_episodes 100 \
--seed 42 &
sleep 2
let "gpu=(gpu+1)%$NUM_GPUS"
done
done
done