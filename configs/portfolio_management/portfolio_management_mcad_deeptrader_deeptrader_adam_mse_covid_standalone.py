task_name = "portfolio_management"
dataset_name = "mcad"
net_name = "deeptrader"
agent_name = "deeptrader"
optimizer_name = "adam"
loss_name = "mse"
regime_name = "covid"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}_{regime_name}"

data = dict(
    type="PortfolioManagementDataset",
    data_path="data/portfolio_management/mcad/covid",
    train_path="data/portfolio_management/mcad/covid/train.csv",
    valid_path="data/portfolio_management/mcad/covid/valid.csv",
    test_path="data/portfolio_management/mcad/covid/test.csv",
    test_dynamic_path="data/portfolio_management/mcad/covid/test_with_label.csv",
    tech_indicator_list=[
        "high", "low", "open", "close", "adjcp", "zopen", "zhigh", "zlow",
        "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25", "zd_30"
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,
)

environment = dict(type="PortfolioManagementDeepTraderEnvironment")

agent = dict(
    type="PortfolioManagementDeepTrader",
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500,
)

trainer = dict(
    type="PortfolioManagementDeepTraderTrainer",
    epochs=200,
    work_dir=work_dir,
    if_remove=False,
)

loss = dict(type="MSELoss")
optimizer = dict(type="Adam", lr=0.001)

# MCAD-specific: Features averaged across assets, so 16 market-level features
act = dict(
    type="AssetScoringNet",
    N=5,  # MCAD has 5 assets: spy, qqq, dbc, agg, gld
    K_l=10,
    num_inputs=16,  # 16 market-level features (averaged across assets)
    num_channels=[12, 12, 12],
    kernel_size=2,
    dropout=0.2,
)

cri = dict(
    type="AssetScoringValueNet",
    N=5,  # MCAD has 5 assets: spy, qqq, dbc, agg, gld
    K_l=10,
    num_inputs=16,  # 16 market-level features (averaged across assets)
    num_channels=[12, 12, 12],
    kernel_size=2,
    dropout=0.2,
)

market = dict(type="MarketScoringNet", n_features=16, hidden_size=12)

transition = dict(type="Transition") 