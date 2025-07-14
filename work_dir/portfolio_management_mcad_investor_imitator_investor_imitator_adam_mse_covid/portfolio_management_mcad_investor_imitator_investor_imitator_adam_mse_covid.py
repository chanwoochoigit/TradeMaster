data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/mcad/covid',
    train_path='data/portfolio_management/mcad/covid/train.csv',
    valid_path='data/portfolio_management/mcad/covid/valid.csv',
    test_path='data/portfolio_management/mcad/covid/test.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.0001,
    test_dynamic_path=
    'data/portfolio_management/mcad/covid/test_with_label.csv',
    test_dynamic='-1')
environment = dict(type='PortfolioManagementInvestorImitatorEnvironment')
agent = dict(
    type='PortfolioManagementInvestorImitator',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)
trainer = dict(
    type='PortfolioManagementInvestorImitatorTrainer',
    epochs=200,
    work_dir=
    'work_dir/portfolio_management_mcad_investor_imitator_investor_imitator_adam_mse_covid',
    if_remove=False)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
act = dict(type='MLPCls', input_dim=110, dims=[128], output_dim=5)
task_name = 'portfolio_management'
dataset_name = 'mcad'
net_name = 'investor_imitator'
agent_name = 'investor_imitator'
optimizer_name = 'adam'
loss_name = 'mse'
regime_name = 'covid'
work_dir = 'work_dir/portfolio_management_mcad_investor_imitator_investor_imitator_adam_mse_covid'
_regime_dates = dict(
    train_start='2006-09-06',
    train_end='2017-07-01',
    val_start='2017-07-01',
    val_end='2020-01-01',
    test_start='2020-01-01',
    test_end='2022-09-01')
train_start_date = '2006-09-06'
val_start_date = '2017-07-01'
test_start_date = '2020-01-01'
test_end_date = '2022-09-01'
