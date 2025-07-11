data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/mcad/trade_war_i',
    train_path='data/portfolio_management/mcad/trade_war_i/train.csv',
    valid_path='data/portfolio_management/mcad/trade_war_i/valid.csv',
    test_path='data/portfolio_management/mcad/trade_war_i/test.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15',
        'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    test_dynamic_path=
    'data/portfolio_management/mcad/trade_war_i/test_with_label.csv')
environment = dict(type='PortfolioManagementEnvironment')
trainer = dict(
    type='PortfolioManagementTrainer',
    agent_name='ppo',
    if_remove=False,
    configs=dict(framework='tf2', num_workers=0),
    work_dir=
    'work_dir/portfolio_management_mcad_ppo_ppo_adam_mse_fg_trade_war_i',
    epochs=2)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
task_name = 'portfolio_management'
dataset_name = 'mcad'
net_name = 'ppo'
agent_name = 'ppo'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_mcad_ppo_ppo_adam_mse_fg_trade_war_i'
