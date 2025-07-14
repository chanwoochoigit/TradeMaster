task_name = "portfolio_management"
dataset_name = "mcad"
net_name = "eiie"
agent_name = "eiie"
optimizer_name = "adam"
loss_name = "mse"
regime_name = "trade_war_i"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}_{regime_name}"

# Import preset regime dates
from configs.regime_dates import get_regime_dates
_regime_dates = get_regime_dates(regime_name)

# Dataset date ranges
train_start_date = _regime_dates["train_start"]
val_start_date = _regime_dates["val_start"]
test_start_date = _regime_dates["test_start"]
test_end_date = _regime_dates["test_end"]

# Transition shape configuration
transition_shape = dict(
    state=dict(shape=(1, 5, 10, 20), type="float32"),
    action=dict(shape=(1, 6), type="float32"),
    reward=dict(shape=(1,), type="float32"),
    done=dict(shape=(1,), type="float32"),
    next_state=dict(shape=(1, 5, 10, 20), type="float32"),
)

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/eiie_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py",
]

data = dict(
    type="PortfolioManagementDataset",
    root=None,
    data_path="data/portfolio_management/mcad/trade_war_i",
    train_path="data/portfolio_management/mcad/trade_war_i/train.csv",
    valid_path="data/portfolio_management/mcad/trade_war_i/valid.csv",
    test_path="data/portfolio_management/mcad/trade_war_i/test.csv",
    test_dynamic_path="data/portfolio_management/mcad/trade_war_i/test_with_label.csv",
    stocks_path="data/portfolio_management/mcad/trade_war_i/stocks.txt",
    aux_stocks_path="data/portfolio_management/mcad/trade_war_i/aux_stocks_files",
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
    type="PortfolioManagementEIIEEnvironment",
    mode="train",
    if_norm=True,
    if_norm_temporal=False,
    scaler=None,
    days=10,
    start_date=None,
    end_date=None,
    initial_amount=1e5,
    transaction_cost_pct=1e-3,
)
transition = dict(type="Transition")
agent = dict(
    type="PortfolioManagementEIIE",
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500,
)

trainer = dict(
    type="PortfolioManagementEIIETrainer", epochs=200, work_dir=work_dir, if_remove=False
)

loss = dict(type="MSELoss")

optimizer = dict(type="Adam", lr=0.001)

act = dict(
    type="EIIEConv",
    input_dim=None,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims=[32],
)

cri = dict(
    type="EIIECritic",
    input_dim=None,
    action_dim=None,
    output_dim=1,
    time_steps=None,
    num_layers=1,
    hidden_size=32,
) 