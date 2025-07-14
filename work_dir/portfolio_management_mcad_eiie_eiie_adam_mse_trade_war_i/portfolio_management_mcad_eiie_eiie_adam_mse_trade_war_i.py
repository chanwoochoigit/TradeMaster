data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/mcad/trade_war_i',
    train_path='data/portfolio_management/mcad/trade_war_i/train.csv',
    valid_path='data/portfolio_management/mcad/trade_war_i/valid.csv',
    test_path='data/portfolio_management/mcad/trade_war_i/test.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    root=None,
    test_dynamic_path=
    'data/portfolio_management/mcad/trade_war_i/test_with_label.csv',
    stocks_path='data/portfolio_management/mcad/trade_war_i/stocks.txt',
    aux_stocks_path=
    'data/portfolio_management/mcad/trade_war_i/aux_stocks_files',
    features_name=[
        'open', 'high', 'low', 'close', 'adjcp', 'volume', 'zopen', 'zhigh',
        'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25',
        'zd_30'
    ],
    temporals_name=['weekday', 'day', 'month'],
    labels_name=['ret1', 'mov1'],
    test_dynamic='-1')
environment = dict(
    type='PortfolioManagementEIIEEnvironment',
    mode='train',
    if_norm=True,
    if_norm_temporal=False,
    scaler=None,
    days=10,
    start_date=None,
    end_date=None,
    initial_amount=100000.0,
    transaction_cost_pct=0.001)
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)
trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=200,
    work_dir=
    'work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i',
    if_remove=False)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
act = dict(
    type='EIIEConv',
    input_dim=16,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims=[32])
cri = dict(
    type='EIIECritic',
    input_dim=16,
    action_dim=5,
    output_dim=1,
    time_steps=10,
    num_layers=1,
    hidden_size=32)
transition = dict(type='Transition')
task_name = 'portfolio_management'
dataset_name = 'mcad'
net_name = 'eiie'
agent_name = 'eiie'
optimizer_name = 'adam'
loss_name = 'mse'
regime_name = 'trade_war_i'
work_dir = 'work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i'
_regime_dates = dict(
    train_start='2006-09-06',
    train_end='2015-07-01',
    val_start='2015-07-01',
    val_end='2018-01-01',
    test_start='2018-01-01',
    test_end='2020-01-01')
train_start_date = '2006-09-06'
val_start_date = '2015-07-01'
test_start_date = '2018-01-01'
test_end_date = '2020-01-01'
transition_shape = dict(
    state=dict(shape=(1, 5, 10, 20), type='float32'),
    action=dict(shape=(1, 6), type='float32'),
    reward=dict(shape=(1, ), type='float32'),
    done=dict(shape=(1, ), type='float32'),
    next_state=dict(shape=(1, 5, 10, 20), type='float32'))
