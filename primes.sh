
run_name="primes-init"

torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_primes \
  run_name="${run_name}"
