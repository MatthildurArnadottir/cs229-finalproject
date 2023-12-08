import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def convert_to_floats(feature_col, memory):
    for i in range(len(feature_col)):
        if (feature_col[i] == "?" or feature_col[i] == "\n"):
            feature_col[i] = memory[i]
        else:
            feature_col[i] = float(feature_col[i])
            memory[i] = feature_col[i]
    return feature_col, memory


def load_crappy_formated_csv_ir(data_path, number_of_cols, pca=False):
    all_x = []
    dates = []
    with open(data_path, "r") as f:
        lineno = -1
        memory = [i for i in range(number_of_cols+1)]
        for line in f:
            lineno += 1
            if (lineno == 0) and not pca:
                continue
            if data_path == "data/energy_data/combined.txt":
                arr = line.split(",")
            else:
                arr = line.split(";")
            if (len(arr) < number_of_cols+1):
                continue
            if pca:
                feature_col = arr
            else:
                feature_col = arr[1:]
            dates.append(arr[0][:7])
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x.append(np.array(feature_col, dtype=np.float32))

    all_x = np.stack(all_x, axis=0)
    all_x -= np.mean(all_x, axis=0)  # normalize
    all_x /= np.std(all_x, axis=0)  # normalize

    all_y = all_x[:, 0].reshape([-1, 1])
    all_x = all_x[:, 1:]

    return all_x, all_y, dates


def load_dates():
    all_dates = []
    with open("data/interest_rate_data/interest_rate_data_all.txt", "r") as f:
        lineno = -1
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(";")
            if (len(arr) < 8):
                continue
            feature_col = arr[0]
            all_dates.append(np.array(feature_col))

    all_dates = np.stack(all_dates, axis=0)
    return all_dates


def cut_in_sequences(x, y, seq_len, inc=1):
    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class InterestRateData:

    def __init__(self, seq_len, data_path, number_of_cols, pca):
        x, y, dates = load_crappy_formated_csv_ir(data_path, number_of_cols, pca)
        self.train_x, self.train_y = cut_in_sequences(x, y, seq_len, inc=seq_len)

        print("train_x.shape:", str(self.train_x.shape))
        print("train_y.shape:", str(self.train_y.shape))

        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size:valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size:valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size:]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size:]]


    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield (batch_x, batch_y)


class InterestRateModel:

    def __init__(self, model_type, model_size, number_of_cols, learning_rate=0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, number_of_cols])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        self.best_stats = 0
        self.model_size = model_size
        head = self.x
        if (model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type.startswith("ltc")):
            learning_rate = 0.01  # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if (model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif (model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head, _ = tf.nn.dynamic_rnn(self.wm, head, dtype=tf.float32, time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif (model_type == "node"):
            self.fused_cell = NODE(model_size, cell_clip=10)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size, cell_clip=-1)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size, cell_clip=-1, global_feedback=True)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        # target_y = tf.expand_dims(self.target_y,axis=-1)
        self.y = tf.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.TruncatedNormal())(head)
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(self.target_y - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.abs(self.target_y - self.y))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results", "economic", "{}_{}.csv".format(model_type, model_size))
        if (not os.path.exists("results/economic")):
            os.makedirs("results/economic")
        if (not os.path.isfile(self.result_file)):
            with open(self.result_file, "w") as f:
                f.write("best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n")

        self.checkpoint_path = os.path.join("tf_sessions", "economic", "{}".format(model_type))
        if (not os.path.exists("tf_sessions/economic")):
            os.makedirs("tf_sessions/economic")

        self.saver = tf.train.Saver()
        self.loss_tracked = []
        self.preds = []

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, economic_data, epochs, verbose=True, log_period=50):

        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        for e in range(epochs):
            if (verbose and e % log_period == 0):
                print("self.target_y:    ", str(self.target_y.shape))
                print("interest_rate_data.test_y ", str(economic_data.test_y.shape))
                print("interest_rate_data.test_x ", str(economic_data.test_x.shape))
                test_acc, test_loss, pred = self.sess.run([self.accuracy, self.loss, self.y],
                                                    {self.x: economic_data.test_x, self.target_y: economic_data.test_y})

                valid_acc, valid_loss = self.sess.run([self.accuracy, self.loss],
                                                      {self.x: economic_data.valid_x, self.target_y: economic_data.valid_y})
                # MSE metric -> less is better
                if ((valid_loss < best_valid_loss and e > 0) or e == 1):
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses), np.mean(accs),
                        valid_loss, valid_acc,
                        test_loss, test_acc
                    )
                    self.save()
                    self.preds.append(pred)

            losses = []
            accs = []
            for batch_x, batch_y in economic_data.iterate_train(batch_size=16):
                acc, loss, _ = self.sess.run([self.accuracy, self.loss, self.train_step],
                                             {self.x: batch_x, self.target_y: batch_y})
                if (not self.constrain_op is None):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            self.loss_tracked.append(losses)
            if (verbose and e % log_period == 0):
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses), np.mean(accs),
                        valid_loss, valid_acc,
                        test_loss, test_acc
                    ))
            if (e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = best_valid_stats
        self.best_stats = best_valid_stats[3]
        print(
            "Best epoch {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                best_epoch,
                train_loss, train_acc,
                valid_loss, valid_acc,
                test_loss, test_acc
            ))
        with open(self.result_file, "a") as f:
            f.write("{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                best_epoch,
                train_loss, train_acc,
                valid_loss, valid_acc,
                test_loss, test_acc
            ))

def plot_losses():

    linear_loss = np.loadtxt('loss_txt_files/linear_loss.txt')
    ctrnn_loss = np.loadtxt('loss_txt_files/ctrnn_loss.txt')
    lstm_loss = np.loadtxt('loss_txt_files/lstm_loss.txt')
    ltc_loss = np.loadtxt('loss_txt_files/ltc_loss.txt')

    plt.plot(linear_loss,label="Linear regression")
    plt.plot(ctrnn_loss, label = "CTRNN")
    plt.plot(lstm_loss, label="LSTM")
    plt.plot(ltc_loss, label="LTC")
    plt.yscale('log')
    plt.legend()
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.show()

def grid_search():
    algos = ['ltc']
    epochs = [100, 200, 300, 400]
    size = [32, 64, 128]

    models = []
    i = 0
    BEST_RESULT = 1000
    best_params = {'epochs': 0, 'num_units': 0}
    for algo in algos:
        for epoch_num in epochs:
            for num_units in size:
                tf.reset_default_graph()
                models.append(InterestRateModel(model_type=algo, model_size=num_units, number_of_cols=number_of_cols))
                models[i].fit(interest_rate_data, epochs=epoch_num, log_period=25)
                print("**********************************************")
                print("MODEL TRAINING COMPLETE num. " + str(i))
                if models[i].best_stats < BEST_RESULT:
                    print("Model " + str(i) + " with")
                    print("Epochs: " + str(epoch_num))
                    print("Num units: " + str(num_units))
                    print("is winning with val loss: " + str(models[i].best_stats))
                    BEST_RESULT = models[i].best_stats
                    best_params['epochs'] = epoch_num
                    best_params['num_units'] = num_units
                print("**********************************************")
                i += 1
    print("Best results: ")
    print(best_params)


if __name__ == "__main__":

    plot_losses()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="lstm")
    parser.add_argument('--log', default=1, type=int)
    parser.add_argument('--size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    args = parser.parse_args()
    seq_len = 32

    # Settings for which dataset to use:
    pca = True
    data_path = "data/interest_rate_data/interest_rate_data_all.txt"
    # data_path = "data/interest_rate_data/pca_data_3.txt"
    number_of_cols = 0
    if data_path == "data/interest_rate_data/interest_rate_data_all.txt":
        number_of_cols = 7
    elif data_path == "data/interest_rate_data/pca_data_3.txt":
        number_of_cols = 3

    interest_rate_data = InterestRateData(seq_len, data_path, number_of_cols, pca)

    model = InterestRateModel(model_type=args.model, model_size=64, number_of_cols=number_of_cols)
    model.fit(interest_rate_data, epochs=args.epochs, log_period=args.log)

    # Visualize results
    x, y, dates = load_crappy_formated_csv_ir(data_path, number_of_cols, pca)
    train_x, train_y = cut_in_sequences(x, y, seq_len, inc=seq_len)
    total_seqs = train_x.shape[1]

    permutation = np.random.RandomState(23489).permutation(total_seqs)
    valid_size = int(0.1 * total_seqs)
    test_size = int(0.15 * total_seqs)
    test_permutation = permutation[valid_size:valid_size + test_size]

    # 3 test sequences
    test_y = np.array(interest_rate_data.test_y)
    predictions = np.array(model.preds)
    absolute_best_predictions = predictions[-1,:,:,:]
    comb_seq1_true = np.concatenate([train_y[:,test_permutation[0]-1,0],train_y[:,test_permutation[0],0]])
    comb_seq1_pred = np.concatenate([train_y[:,test_permutation[0]-1,0],absolute_best_predictions[:,0,0]])
    comb_seq1_true = np.concatenate([train_y[:,test_permutation[0]-2,0],comb_seq1_true])
    comb_seq1_pred = np.concatenate([train_y[:,test_permutation[0]-2,0],comb_seq1_pred])

    comb_seq2_true = np.concatenate([train_y[:, test_permutation[1] - 1, 0], train_y[:, test_permutation[1], 0]])
    comb_seq2_pred = np.concatenate([train_y[:, test_permutation[1] - 1, 0], absolute_best_predictions[:, 1, 0]])

    comb_seq3_true = np.concatenate([train_y[:,test_permutation[2]-1,0],train_y[:,test_permutation[2],0]])
    comb_seq3_pred = np.concatenate([train_y[:,test_permutation[2]-1,0],absolute_best_predictions[:,2,0]])
    comb_seq3_true = np.concatenate([train_y[:,test_permutation[2]-2,0],comb_seq3_true])
    comb_seq3_pred = np.concatenate([train_y[:,test_permutation[2]-2,0],comb_seq3_pred])

    all_dates = load_dates()
    sequenced_dates = []
    inc = 1
    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequenced_dates.append(all_dates[start:end])

    seq1_dates = []
    for i in range(test_permutation[0]-2,test_permutation[0]+1):
        for j in range(seq_len):
            seq1_dates.append(sequenced_dates[i][j])

    seq2_dates = []
    for i in range(test_permutation[1]-1,test_permutation[1]+1):
        for j in range(seq_len):
            seq2_dates.append(sequenced_dates[i][j])

    seq3_dates = []
    for i in range(test_permutation[2]-2,test_permutation[2]+1):
        for j in range(seq_len):
            seq3_dates.append(sequenced_dates[i][j])

    seq1_dates_idx = [i for i in range(0, seq_len*3, 8)]
    seq1_dates_few = [seq1_dates[i] for i in range(0, seq_len*3, 8)]

    seq2_dates_idx = [i for i in range(0, seq_len*2, 8)]
    seq2_dates_few = [seq2_dates[i] for i in range(0, seq_len*2, 8)]

    seq3_dates_idx = [i for i in range(0, seq_len*3, 8)]
    seq3_dates_few = [seq3_dates[i] for i in range(0, seq_len*3, 8)]

    plt.figure(figsize=(7,5),tight_layout=True)
    plt.plot(comb_seq1_pred,label="Predicted")
    plt.plot(comb_seq1_true,label="True")
    plt.xticks(seq1_dates_idx, seq1_dates_few, rotation = 45, fontsize=8)
    plt.ylabel("Normalized Interest Rate")
    plt.legend()
    plt.title('CT-RNN - test sequence 1')
    plt.show()

    plt.figure(figsize=(7,5),tight_layout=True)
    plt.plot(comb_seq2_pred,label="Predicted")
    plt.plot(comb_seq2_true,label="True")
    plt.xticks(seq2_dates_idx, seq2_dates_few, rotation = 45, fontsize=8)
    plt.ylabel("Normalized Interest Rate")
    plt.legend()
    plt.title('CT-RNN - test sequence 2')
    plt.show()

    plt.figure(figsize=(7,5),tight_layout=True)
    plt.plot(comb_seq3_pred,label="Predicted")
    plt.plot(comb_seq3_true,label="True")
    plt.xticks(seq3_dates_idx, seq3_dates_few, rotation = 45, fontsize=8)
    plt.ylabel("Normalized Interest Rate")
    plt.legend()
    plt.title('LTC - test sequence 3')
    plt.show()



