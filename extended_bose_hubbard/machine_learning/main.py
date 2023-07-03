import train_model as tm

data_info = [
    {
        'data_title': 'lambda',
        'train_data_path': '../data/multiclass_lambda/multiclass_lambda_train.csv',
        'test_data_path': '../data/multiclass_lambda/multiclass_lambda_test.csv',
    },
    {
        'data_title': 'lambda_gamma',
        'train_data_path': '../data/multiclass_lambda_gamma/multiclass_lambda_gamma_train.csv',
        'test_data_path': '../data/multiclass_lambda_gamma/multiclass_lambda_gamma_test.csv',
    },
    {
        'data_title': 'mps',
        'train_data_path': '../data/multiclass_mps/multiclass_mps_train.csv',
        'test_data_path': '../data/multiclass_mps/multiclass_mps_test.csv',
    },
]

for i in range(len(data_info)):
    obj = tm.TrainModel(data_title=data_info[i]['data_title'],
                        train_data_path=data_info[i]['train_data_path'],
                        test_data_path=data_info[i]['test_data_path'],
                        n_node_hidden_layer=100,
                        n_classes=3,
                        n_epoch=200,
                        lr=0.001,
                        batch_size=10,
                        weight_decay=0.001)

    # obj.train_model()
    # obj.test_model()
    obj.plot_predict()
