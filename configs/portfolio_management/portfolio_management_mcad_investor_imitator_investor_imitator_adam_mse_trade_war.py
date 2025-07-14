task_name = "portfolio_management"
dataset_name = "mcad"
net_name = "investor_imitator"
agent_name = "investor_imitator"
optimizer_name = "adam"
loss_name = "mse"
regime_name = "trade_war"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}_{regime_name}"

# Import preset regime dates
from configs.regime_dates import get_regime_dates
_regime_dates = get_regime_dates(regime_name)

# Dataset date ranges
train_start_date = _regime_dates["train_start"]
val_start_date = _regime_dates["val_start"]
test_start_date = _regime_dates["test_start"]
test_end_date = _regime_dates["test_end"]

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/investor_imitator_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
]

data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/mcad/trade_war',
    train_path='data/portfolio_management/mcad/trade_war/train.csv',
    valid_path='data/portfolio_management/mcad/trade_war/valid.csv',
    test_path='data/portfolio_management/mcad/trade_war/test.csv',
    test_dynamic_path='data/portfolio_management/mcad/trade_war/test_with_label.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.0001
)

environment = dict(type='PortfolioManagementInvestorImitatorEnvironment')

agent = dict(
    type='PortfolioManagementInvestorImitator',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500
)

trainer = dict(
    type='PortfolioManagementInvestorImitatorTrainer',
    epochs=200,
    work_dir=work_dir,
    if_remove=False
)

loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)

act = dict(
    type='MLPCls',
    input_dim = None,
    dims = [128],
    output_dim = None
) 