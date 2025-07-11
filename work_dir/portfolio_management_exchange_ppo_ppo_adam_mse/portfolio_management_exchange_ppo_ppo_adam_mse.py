data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/exchange',
    train_path='data/portfolio_management/exchange/train.csv',
    valid_path='data/portfolio_management/exchange/valid.csv',
    test_path='data/portfolio_management/exchange/test.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15',
        'zd_20', 'zd_25', 'zd_30'
    ],
    initial_amount=100000,
    transaction_cost_pct=0.001,
    test_dynamic_path=
    'data/portfolio_management/exchange/test_labeled_3_24_-0.05_0.05.csv')
environment = dict(type='PortfolioManagementEnvironment')
trainer = dict(
    type='PortfolioManagementTrainer',
    agent_name='ppo',
    if_remove=False,
    configs=dict(framework='tf2', num_workers=0),
    work_dir='work_dir/portfolio_management_exchange_ppo_ppo_adam_mse',
    epochs=10)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
task_name = 'portfolio_management'
dataset_name = 'exchange'
net_name = 'ppo'
agent_name = 'ppo'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_exchange_ppo_ppo_adam_mse'
