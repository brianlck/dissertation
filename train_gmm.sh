
#!/bin/bash
declare -a bridge=(32 128 256)


for n_bridge in "${bridge[@]}"
do
python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel false --loss_fn traj-balance
done

for n_bridge in "${bridge[@]}"
do
python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel false  --loss_fn log-var


python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn log-var --repel-percentage 0.5

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn log-var --repel-percentage 1.0

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel false  --loss_fn subtraj-log-var

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn subtraj-log-var --repel-percentage 0.5

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn subtraj-log-var --repel-percentage 1.0
done


for n_bridge in "${bridge[@]}"
do
python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel false  --loss_fn log-var

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn log-var --repel-percentage 0.5

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn log-var --repel-percentage 1.0

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel false  --loss_fn subtraj-log-var

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn subtraj-log-var --repel-percentage 0.5

python3 main.py --x_dim 2 --t_dim 38 --h_dim 40 --n_bridges $n_bridge --target gmm --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn subtraj-log-var --repel-percentage 1.0
done


for n_bridge in "${bridge[@]}"
do
python3 main.py --x_dim 10 --t_dim 48 --h_dim 58 --n_bridges $n_bridge --target funnel --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel false  --loss_fn log-var

python3 main.py --x_dim 10 --t_dim 48 --h_dim 58 --n_bridges $n_bridge --target funnel --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn log-var --repel-percentage 0.5

python3 main.py --x_dim 10 --t_dim 48 --h_dim 58 --n_bridges $n_bridge --target funnel --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn log-var --repel-percentage 1.0

python3 main.py --x_dim 10 --t_dim 48 --h_dim 58 --n_bridges $n_bridge --target funnel --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel false  --loss_fn subtraj-log-var

python3 main.py --x_dim 10 --t_dim 48 --h_dim 58 --n_bridges $n_bridge --target funnel --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn subtraj-log-var --repel-percentage 0.5

python3 main.py --x_dim 10 --t_dim 48 --h_dim 58 --n_bridges $n_bridge --target funnel --anneal geometric --max_epoch 30000 --batch_size 5000 --init_eps 0.01 --eps_trainable False --lr 0.005 --seed 5 --correct false --repel true  --loss_fn subtraj-log-var --repel-percentage 1.0
done