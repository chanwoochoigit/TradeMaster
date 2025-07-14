# base parameters (do not modify)
root = None
workdir = "workdir"
tag = "mask_sac_mcad_covid"
num_stocks = 5  # MCAD has 5 stocks: spy, qqq, dbc, agg, gld
num_envs = 1
num_features = 20  # MCAD features (17 tech indicators + 3 temporals)
temporal_dim = 3 # weekday, day, month
train_start_date = "2006-09-06"
val_start_date = "2017-07-01"
test_start_date = "2020-01-01"
test_end_date = None
if_use_per = False
if_use_rep = True
if_use_beta = True
if_norm = True
if_norm_temporal = False
save_freq = 20
repeat_times = 128
action_wrapper_method = "reweight"
T = 1.0
n_steps_per_episode = 1024

# train parameters (adjust mainly)
num_episodes = 200
days = 10
batch_size = 64
buffer_size = 4096
horizon_len = 64
embed_dim = 128
decoder_embed_dim = 128
depth = 1 # 1 transformer
decoder_depth = 1
lr = 5e-5 # act_lr, cri_lr
act_lr = 5e-5
cri_lr = 5e-5
rep_lr = 5e-5
beta_lr = 5e-5
rep_loss_weight = 0.01
beta_loss_weight = 0.01
seed = 10

# size
feature_size = (days, num_features)
patch_size = (days, num_features)

transition = ["state", "action", "mask", "ids_restore", "reward", "done", "next_state"]
transition_shape = dict(
    state=dict(shape=(num_envs, num_stocks, days, num_features), type="float32"),
    action=dict(shape=(num_envs, num_stocks + 1), type="float32"),
    mask=dict(shape=(num_envs, num_stocks), type="int32"),
    ids_restore=dict(shape=(num_envs, num_stocks), type="int64"),
    reward=dict(shape=(num_envs,), type="float32"),
    done=dict(shape=(num_envs,), type="float32"),
    next_state=dict(shape=(num_envs, num_stocks, days, num_features), type="float32"),
)

dataset = dict(
    type="PortfolioManagementDataset",
    root=root,
    data_path="data/portfolio_management/mcad/covid",
    stocks_path="data/portfolio_management/mcad/covid/stocks.txt",
    aux_stocks_path="data/portfolio_management/mcad/covid/aux_stocks_files",
    features_name=[
        "open",
        "high",
        "low",
        "close",
        "adjcp",
        "volume",
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30",
    ],
    temporals_name=[
        "weekday",
        "day",
        "month",
    ],
    labels_name=[
        "ret1",
        "mov1",
    ],
)

environment = dict(
    type="EnvironmentPV",
    dataset=None,
    mode="train",
    if_norm=if_norm,
    if_norm_temporal=if_norm_temporal,
    scaler=None,
    days=days,
    start_date=None,
    end_date=None,
    initial_amount=1e3,
    transaction_cost_pct=1e-3,
)

rep_net = dict(
    type="MaskTimeState",
    embed_type="TimesEmbed",
    feature_size=feature_size,
    patch_size=patch_size,
    t_patch_size=1,
    num_stocks=num_stocks,
    pred_num_stocks=num_stocks,
    in_chans=1,
    input_dim=num_features,
    temporal_dim=temporal_dim,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=4,
    decoder_embed_dim=decoder_embed_dim,
    decoder_depth=decoder_depth,
    decoder_num_heads=8,
    mlp_ratio=4.0,
    norm_pix_loss=False,
    cls_embed=True,
    sep_pos_embed=True,
    trunc_init=False,
    no_qkv_bias=False,
    mask_ratio_min=0.6,
    mask_ratio_max=0.8,
    mask_ratio_mu=0.7,
    mask_ratio_std=0.1,
)

act_net = dict(
    type="ActorMaskSAC",
    embed_dim=decoder_embed_dim,
    depth=depth,
    cls_embed=True,
)

cri_net = dict(
    type="CriticMaskSAC",
    embed_dim=decoder_embed_dim,
    depth=depth,
    cls_embed=True,
)

criterion = dict(type="MSELoss", reduction="none")
scheduler = dict(
    type="MultiStepLRScheduler",
    multi_steps=[
        120 * n_steps_per_episode,
        200 * n_steps_per_episode,
        280 * n_steps_per_episode,
    ],
    t_initial=num_episodes * n_steps_per_episode,
    decay_t=500 * n_steps_per_episode,
    gamma=0.1,
    t_mul=1.0,
    lr_min=0.0,
    decay_rate=1.0,
    warmup_t=60 * n_steps_per_episode,
    warmup_lr_init=1e-8,
    warmup_prefix=False,
    cycle_limit=0,
    t_in_epochs=False,
    noise_range_t=None,
    noise_pct=0.67,
    noise_std=1.0,
    noise_seed=42,
    initialize=True,
)
optimizer = dict(type="AdamW", params=None, lr=lr)

agent = dict(
    type="AgentMaskSAC",
    act_lr=act_lr,
    cri_lr=cri_lr,
    rep_lr=rep_lr,
    beta_lr=beta_lr,
    rep_net=rep_net,
    act_net=act_net,
    cri_net=cri_net,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    if_use_per=if_use_per,
    if_use_rep=if_use_rep,
    if_use_beta=if_use_beta,
    rep_loss_weight=rep_loss_weight,
    beta_loss_weight=beta_loss_weight,
    num_envs=num_envs,
    transition_shape=transition_shape,
    max_step=1e4,
    gamma=0.99,
    reward_scale=2**0,
    repeat_times=repeat_times,
    batch_size=batch_size,
    clip_grad_norm=3.0,
    soft_update_tau=5e-3,
    state_value_tau=0,
    device=None,
    action_wrapper_method=action_wrapper_method,
    T=T,
)
