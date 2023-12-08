import numpy as np
import matplotlib.pyplot as plt
import keras

from interest_rate import load_crappy_formated_csv_ir
from interest_rate import load_dates
from interest_rate import cut_in_sequences
from energy import load_crappy_formated_csv_energy
from energy import load_pca_data
from combined_datasets import load_pca_combined

def fit_linear_model_interest_rate(data_path, pca):
    number_of_cols = 0
    if data_path == "data/interest_rate_data/interest_rate_data_all.txt":
        number_of_cols = 7
    elif data_path == "data/interest_rate_data/pca_data_3.txt":
        number_of_cols = 3

    X, y, dates = load_crappy_formated_csv_ir(data_path, number_of_cols, pca)
    seq_len = 32
    train_x, train_y = cut_in_sequences(X, y, 32, 32)
    print("train_x.shape:", str(train_x.shape))
    print("train_y.shape:", str(train_y.shape))

    total_seqs = train_x.shape[1]
    print("Total number of training sequences: {}".format(total_seqs))
    permutation = np.random.RandomState(23489).permutation(total_seqs)

    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    test_x = train_x[:, permutation[valid_size:valid_size + test_size]]
    test_y = train_y[:, permutation[valid_size:valid_size + test_size]]
    train_x2 = train_x[:, permutation[valid_size + test_size:]]
    train_y2 = train_y[:, permutation[valid_size + test_size:]]

    test_permutation = permutation[valid_size:valid_size + test_size]

    all_train_x = np.concatenate([train_x2[:, 0, :], train_x2[:, 1, :]],axis=0)
    all_train_y = np.concatenate([train_y2[:, 0, 0], train_y2[:, 1, 0]],axis=0)
    for i in range(5,train_x.shape[1]):
        all_train_x = np.concatenate([all_train_x, train_x[:, i, :]],axis=0)
        all_train_y = np.concatenate([all_train_y, train_y[:, i, 0]],axis=0)

    flattened_train_x = np.zeros((train_x2.shape[0]*train_x2.shape[2], train_x2.shape[1]))
    flattened_train_y = np.zeros((train_y2.shape[0],train_y2.shape[1]))
    for i in range(train_x2.shape[1]):
        flattened_train_x[:,i] = train_x2[:,i,:].reshape((train_x2.shape[0]*train_x2.shape[2],))
        flattened_train_y[:,i] = train_y2[:,i,0]

    flattened_test_x = np.zeros((test_x.shape[0] * test_x.shape[2], test_x.shape[1]))
    flattened_test_y = np.zeros((test_y.shape[0], test_y.shape[1]))
    for i in range(test_x.shape[1]):
        flattened_test_x[:, i] = test_x[:, i, :].reshape((test_x.shape[0]*test_x.shape[2],))
        flattened_test_y[:,i] = test_y[:, i, 0]

    print("input X shape to model:", str(flattened_test_x.shape))
    print("input y shape to model:", str(flattened_test_y.shape))
    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(32, input_shape=(flattened_test_x.shape[0],))
    ])

    # Compile the model
    model.compile(loss='mse', optimizer='SGD')

    # Fit the model
    history = model.fit(flattened_train_x.T, flattened_train_y.T, epochs=200, batch_size=16)
    np.savetxt('linear_loss.txt', history.history['loss'])

    # Evaluate the model
    preds = model.predict(flattened_test_x.T)
    preds = preds.T

    print("MSE= "+str(np.mean(np.square(preds-flattened_test_y))))
    comb_seq1_true = np.concatenate([train_y[:, test_permutation[0] - 1, 0], train_y[:, test_permutation[0], 0]])
    comb_seq1_pred = np.concatenate([train_y[:, test_permutation[0] - 1, 0], preds[:,0]])
    comb_seq1_true = np.concatenate([train_y[:, test_permutation[0] - 2, 0], comb_seq1_true])
    comb_seq1_pred = np.concatenate([train_y[:, test_permutation[0] - 2, 0], comb_seq1_pred])

    comb_seq2_true = np.concatenate([train_y[:, test_permutation[1] - 1, 0], train_y[:, test_permutation[1], 0]])
    comb_seq2_pred = np.concatenate([train_y[:, test_permutation[1] - 1, 0], preds[:, 1]])

    comb_seq3_true = np.concatenate([train_y[:,test_permutation[2]-1,0],train_y[:,test_permutation[2],0]])
    comb_seq3_pred = np.concatenate([train_y[:,test_permutation[2]-1,0],preds[:,2]])
    comb_seq3_true = np.concatenate([train_y[:,test_permutation[2]-2,0],comb_seq3_true])
    comb_seq3_pred = np.concatenate([train_y[:,test_permutation[2]-2,0],comb_seq3_pred])

    all_dates = load_dates()
    sequenced_dates = []
    inc = 1
    for s in range(0, X.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequenced_dates.append(all_dates[start:end])

    seq1_dates = []
    for i in range(test_permutation[0] - 2, test_permutation[0] + 1):
        for j in range(seq_len):
            seq1_dates.append(sequenced_dates[i][j])
    seq2_dates = []
    for i in range(test_permutation[1] - 1, test_permutation[1] + 1):
        for j in range(seq_len):
            seq2_dates.append(sequenced_dates[i][j])
    seq3_dates = []
    for i in range(test_permutation[2] - 2, test_permutation[2] + 1):
        for j in range(seq_len):
            seq3_dates.append(sequenced_dates[i][j])

    seq1_dates_idx = [i for i in range(0, seq_len * 3, 8)]
    seq1_dates_few = [seq1_dates[i] for i in range(0, seq_len * 3, 8)]
    seq2_dates_idx = [i for i in range(0, seq_len * 2, 8)]
    seq2_dates_few = [seq2_dates[i] for i in range(0, seq_len * 2, 8)]
    seq3_dates_idx = [i for i in range(0, seq_len * 3, 8)]
    seq3_dates_few = [seq3_dates[i] for i in range(0, seq_len * 3, 8)]

    plt.figure(figsize=(7, 5), tight_layout=True)
    plt.plot(comb_seq1_pred, label="Predicted")
    plt.plot(comb_seq1_true, label="True")
    plt.xticks(seq1_dates_idx, seq1_dates_few, rotation=45, fontsize=8)
    plt.ylabel("Normalized Interest Rate")
    plt.legend()
    plt.title('Linear regression - test sequence 1')
    plt.show()

    plt.figure(figsize=(7,5),tight_layout=True)
    plt.plot(comb_seq2_pred,label="Predicted")
    plt.plot(comb_seq2_true,label="True")
    plt.xticks(seq2_dates_idx, seq2_dates_few, rotation = 45, fontsize=8)
    plt.ylabel("Normalized Interest Rate")
    plt.legend()
    plt.title('Linear regression - test sequence 2')
    plt.show()

    plt.figure(figsize=(7,5),tight_layout=True)
    plt.plot(comb_seq3_pred,label="Predicted")
    plt.plot(comb_seq3_true,label="True")
    plt.xticks(seq3_dates_idx, seq3_dates_few, rotation = 45, fontsize=8)
    plt.ylabel("Normalized Interest Rate")
    plt.legend()
    plt.title('Linear regression - test sequence 3')
    plt.show()

def fit_linear_model_power():

    X, y = load_crappy_formated_csv_energy()
    # X,y = load_pca_data()
    seq_len = 32
    train_x, train_y = cut_in_sequences(X, y, 16, 16)
    print("train_x.shape:", str(train_x.shape))
    print("train_y.shape:", str(train_y.shape))

    total_seqs = train_x.shape[1]
    permutation = np.random.RandomState(23489).permutation(total_seqs)

    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    test_x = train_x[:, permutation[valid_size:valid_size + test_size]]
    test_y = train_y[:, permutation[valid_size:valid_size + test_size]]
    train_x2 = train_x[:, permutation[valid_size + test_size:]]
    train_y2 = train_y[:, permutation[valid_size + test_size:]]

    test_permutation = permutation[valid_size:valid_size + test_size]

    all_train_x = np.concatenate([train_x2[:, 0, :], train_x2[:, 1, :]], axis=0)
    all_train_y = np.concatenate([train_y2[:, 0, 0], train_y2[:, 1, 0]], axis=0)
    for i in range(5, train_x.shape[1]):
        all_train_x = np.concatenate([all_train_x, train_x[:, i, :]], axis=0)
        all_train_y = np.concatenate([all_train_y, train_y[:, i, 0]], axis=0)

    flattened_train_x = np.zeros((train_x2.shape[0] * train_x2.shape[2], train_x2.shape[1]))
    flattened_train_y = np.zeros((train_y2.shape[0], train_y2.shape[1]))
    for i in range(train_x2.shape[1]):
        flattened_train_x[:, i] = train_x2[:, i, :].reshape((train_x2.shape[0] * train_x2.shape[2],))
        flattened_train_y[:, i] = train_y2[:, i, 0]

    flattened_test_x = np.zeros((test_x.shape[0] * test_x.shape[2], test_x.shape[1]))
    flattened_test_y = np.zeros((test_y.shape[0], test_y.shape[1]))
    for i in range(test_x.shape[1]):
        flattened_test_x[:, i] = test_x[:, i, :].reshape((test_x.shape[0] * test_x.shape[2],))
        flattened_test_y[:, i] = test_y[:, i, 0]

    print("input X shape to model:", str(flattened_test_x.shape))
    print("input y shape to model:", str(flattened_test_y.shape))
    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(16, input_shape=(flattened_test_x.shape[0],))
    ])

    # Compile the model
    model.compile(loss='mse', optimizer='SGD')

    # Fit the model
    history = model.fit(flattened_train_x.T, flattened_train_y.T, epochs=400,  verbose=True)
    np.savetxt("linear_loss_en.txt", history.history['loss'])
    # Evaluate the model
    preds = model.predict(flattened_test_x.T)
    preds = preds.T

    print("MSE= " + str(np.mean(np.square(preds - flattened_test_y))))

    # Visualize results
    x, y = load_crappy_formated_csv_energy()
    train_x, train_y = cut_in_sequences(x, y, 16, inc=16)
    total_seqs = train_x.shape[1]
    permutation = np.random.RandomState(23489).permutation(total_seqs)
    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)
    test_permutation = permutation[valid_size:valid_size + test_size]

    print("Preds:")
    print(preds.shape)
    print("Test seq: ")
    print(train_y[:, test_permutation[0], 0].shape)

    # 3 test sequences
    comb_seq1_true = np.concatenate([train_y[:, test_permutation[0] - 1, 0], train_y[:, test_permutation[0], 0]])
    comb_seq1_pred = np.concatenate([train_y[:, test_permutation[0] - 1, 0], preds[:,0]])
    comb_seq1_true = np.concatenate([train_y[:, test_permutation[0] - 2, 0], comb_seq1_true])
    comb_seq1_pred = np.concatenate([train_y[:, test_permutation[0] - 2, 0], comb_seq1_pred])

    # all_dates = load_dates()
    sequenced_dates = []
    seq_len = 16
    inc = 1
    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequenced_dates.append(np.arange(start, end))

    seq1_dates = []
    for i in range(test_permutation[0] - 2, test_permutation[0] + 1):
        for j in range(seq_len):
            seq1_dates.append(sequenced_dates[i][j])

    seq1_dates_idx = [i for i in range(0, len(seq1_dates), 4)]

    plt.figure(figsize=(7, 5), tight_layout=True)
    plt.plot(comb_seq1_pred, label="Predicted")
    plt.plot(comb_seq1_true, label="True")
    plt.xticks(seq1_dates_idx, seq1_dates_idx, rotation=45, fontsize=8)
    plt.ylabel("Normalized Industrial Energy Consumption")
    plt.xlabel("Time (months, index of sequence)")
    plt.legend()
    plt.title('Linear Regression')
    plt.show()

def fit_linear_model_combined():

    X, y = load_pca_combined()
    train_x, train_y = cut_in_sequences(X, y, 16, 16)
    print("train_x.shape:", str(train_x.shape))
    print("train_y.shape:", str(train_y.shape))

    total_seqs = train_x.shape[1]
    permutation = np.random.RandomState(23489).permutation(total_seqs)

    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)

    test_x = train_x[:, permutation[valid_size:valid_size + test_size]]
    test_y = train_y[:, permutation[valid_size:valid_size + test_size]]
    train_x2 = train_x[:, permutation[valid_size + test_size:]]
    train_y2 = train_y[:, permutation[valid_size + test_size:]]

    all_train_x = np.concatenate([train_x2[:, 0, :], train_x2[:, 1, :]], axis=0)
    all_train_y = np.concatenate([train_y2[:, 0, 0], train_y2[:, 1, 0]], axis=0)
    for i in range(5, train_x.shape[1]):
        all_train_x = np.concatenate([all_train_x, train_x[:, i, :]], axis=0)
        all_train_y = np.concatenate([all_train_y, train_y[:, i, 0]], axis=0)

    flattened_train_x = np.zeros((train_x2.shape[0] * train_x2.shape[2], train_x2.shape[1]))
    flattened_train_y = np.zeros((train_y2.shape[0], train_y2.shape[1]))
    for i in range(train_x2.shape[1]):
        flattened_train_x[:, i] = train_x2[:, i, :].reshape((train_x2.shape[0] * train_x2.shape[2],))
        flattened_train_y[:, i] = train_y2[:, i, 0]

    flattened_test_x = np.zeros((test_x.shape[0] * test_x.shape[2], test_x.shape[1]))
    flattened_test_y = np.zeros((test_y.shape[0], test_y.shape[1]))
    for i in range(test_x.shape[1]):
        flattened_test_x[:, i] = test_x[:, i, :].reshape((test_x.shape[0] * test_x.shape[2],))
        flattened_test_y[:, i] = test_y[:, i, 0]

    print("input X shape to model:", str(flattened_test_x.shape))
    print("input y shape to model:", str(flattened_test_y.shape))
    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(16, input_shape=(flattened_test_x.shape[0],))
    ])

    # Compile the model
    model.compile(loss='mse', optimizer='SGD')

    # Fit the model
    history = model.fit(flattened_train_x.T, flattened_train_y.T, epochs=400, verbose=False)
    np.savetxt("linear_loss_comb.txt", history.history['loss'])

    # Evaluate the model
    preds = model.predict(flattened_test_x.T)
    preds = preds.T

    print("MSE= " + str(np.mean(np.square(preds - flattened_test_y))))


if __name__ == "__main__":

    pca = True
    # data_path = "data/interest_rate_data/interest_rate_data_all.txt"
    # data_path = "data/interest_rate_data/pca_data_3.txt"

    # fit_linear_model_interest_rate(data_path, True)
    # fit_linear_model_power()
    fit_linear_model_combined()
