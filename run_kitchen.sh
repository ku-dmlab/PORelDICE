NUM_GPUS=4
ENV_LIST=(
	"kitchen-complete-v0"
	# "kitchen-partial-v0"
	# "kitchen-mixed-v0"
)

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
--config=configs/kitchen_config.py \
--alg "PORelDICE" \
--alpha $alpha \
--epsilon $epsilon \
--max_steps 1000000 \
--log_interval 100 \
--eval_interval 5000 \
--eval_episodes 5 \
--seed 42 &
sleep 2
let "gpu=(gpu+1)%$NUM_GPUS"
done
done
done