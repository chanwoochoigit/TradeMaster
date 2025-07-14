task_name = "portfolio_management"
dataset_name = "mcad"
net_name = "sarl"
agent_name = "sarl"
optimizer_name = "adam"
loss_name = "mse"
regime_name = "trade_war_i"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}_{regime_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/mcad.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/trainers/{task_name}/sarl_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
]

data = dict(
    type="PortfolioManagementDataset",
    data_path="data/portfolio_management/mcad/trade_war_i",
    train_path="data/portfolio_management/mcad/trade_war_i/train.csv",
    valid_path="data/portfolio_management/mcad/trade_war_i/valid.csv",
    test_path="data/portfolio_management/mcad/trade_war_i/test.csv",
    test_dynamic_path="data/portfolio_management/mcad/trade_war_i/test_with_label.csv",
    tech_indicator_list=[
        "high", "low", "open", "close", "adjcp", "zopen", "zhigh", "zlow",
        "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25", "zd_30"
    ],
    length_day=5,
    initial_amount=10000,
    transaction_cost_pct=0.001
)

environment = dict(
    type="PortfolioManagementSARLEnvironment",
)

trainer = dict(
    type="PortfolioManagementSARLTrainer",
    agent_name="ddpg",
    if_remove=False,
    configs={
        "framework": "tf2",
        "num_workers": 0,
    },
    work_dir=work_dir,
    epochs=200,
) 