import train_model as tm

data_info = [
    {
        'data_title': 'lambda',
        'train_data_path': '../data/L4/lambda/lambda_train.csv',
        'test_data_path': '../data/L4/lambda/lambda_test.csv',
    },
    {
        'data_title': 'lambda_gamma',
        'train_data_path': '../data/L4/lambda_gamma/lambda_gamma_train.csv',
        'test_data_path': '../data/L4/lambda_gamma/lambda_gamma_test.csv',
    },
    {
        'data_title': 'mps',
        'train_data_path': '../data/L4/mps/mps_train.csv',
        'test_data_path': '../data/L4/mps/mps_test.csv',
    },
]

for i in range(len(data_info)):
    obj_fc = tm.FullyConnected(data_title=data_info[i]['data_title'],
                               train_data_path=data_info[i]['train_data_path'],
                               test_data_path=data_info[i]['test_data_path'],
                               n_node_hidden_layer=100,
                               n_classes=2,
                               n_epoch=200,
                               lr=0.001,
                               batch_size=10,
                               weight_decay=0.001)

    obj_fc.train_model(loss_func='BCELoss')
    obj_fc.test_model()
    obj_fc.plot_predict()

    # obj_cnn = tm.CNN1D(data_title=data_info[i]['data_title'],
    #                    train_data_path=data_info[i]['train_data_path'],
    #                    test_data_path=data_info[i]['test_data_path'],
    #                    n_node_hidden_layer=100,
    #                    n_classes=3,
    #                    n_epoch=200,
    #                    lr=0.001,
    #                    batch_size=10,
    #                    weight_decay=0.001,
    #                    momentum=0.9)
    #
    # obj_cnn.train_model()
    # obj_cnn.test_model()
    # obj_fc.plot_predict()
