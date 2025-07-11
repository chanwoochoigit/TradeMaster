data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/dj30',
    train_path='data/portfolio_management/dj30/new_train.csv',
    valid_path='data/portfolio_management/dj30/new_valid.csv',
    test_path='data/portfolio_management/dj30/new_test.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15',
        'zd_20', 'zd_25', 'zd_30', 'autoFE_f_0', 'autoFE_f_1', 'autoFE_f_2',
        'autoFE_f_3', 'autoFE_f_4', 'autoFE_f_5', 'autoFE_f_6', 'autoFE_f_7',
        'autoFE_f_8', 'autoFE_f_9'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    test_dynamic_path='data/portfolio_management/dj30/test_with_label.csv')
environment = dict(type='PortfolioManagementEnvironment')
trainer = dict(
    type='PortfolioManagementTrainer',
    agent_name='ppo',
    if_remove=False,
    configs=dict(framework='tf2', num_workers=0),
    work_dir='work_dir/portfolio_management_dj30_ppo_ppo_adam_mse_fg',
    epochs=2)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
task_name = 'portfolio_management'
dataset_name = 'dj30'
net_name = 'ppo'
agent_name = 'ppo'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_dj30_ppo_ppo_adam_mse_fg'
