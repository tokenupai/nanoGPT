# 使用 GPT-2 BPE 分词的莎士比亚数据，从零训练小模型

out_dir = 'out-shakespeare-gpt2'
eval_interval = 250
eval_iters = 40
log_interval = 10

# 小数据集容易过拟合，只在验证损失改善时保存
always_save_checkpoint = False
init_from = 'scratch'

wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt-scratch'

dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256 # 每个输入片段包含 256 个 BPE token

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2
bias = False

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'mps'
dtype = 'float32'
compile = False
