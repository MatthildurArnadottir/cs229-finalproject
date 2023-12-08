import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
from interest_rate import load_crappy_formated_csv_ir


def load_dates():
    all_dates = []
    with open("data/electricity/consumption.txt", "r") as f:
        lineno = -1
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(";")
            if (len(arr) < 15):
                continue
            feature_col = arr[0]
            print(feature_col)
            all_dates.append(np.array(feature_col))

    all_dates = np.stack(all_dates, axis=0)
    return all_dates

def convert_to_floats(feature_col, memory):
    for i in range(len(feature_col)):
        if (feature_col[i] == "?" or feature_col[i] == "\n"):
            feature_col[i] = memory[i]
        else:
            feature_col[i] = float(feature_col[i])
            memory[i] = feature_col[i]
    return feature_col, memory

def load_combined_csv():

    output = []
    with open("data/electricity/consumption.txt", "r") as f:
        lineno = -1
        memory = 0
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            if (len(arr) < 16):
                continue
            feature_col = arr[1]
            #print(feature_col)
            #feature_col, memory = convert_to_floats(feature_col, memory)
            output.append(np.array(feature_col, dtype=np.float32))


    extracted_features = []
    with open("data/electricity/combined.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(10)]
        for line in f:
            lineno += 1
            arr = line.split(",")
            if (len(arr) < 10):
                continue
            feature_col = arr[:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            extracted_features.append(np.array(feature_col, dtype=np.float32))

    output = np.stack(output, axis = 0).reshape((-1,1))
    extracted_features = np.stack(extracted_features, axis = 0)

    all_x = np.concatenate((output, extracted_features), axis = 1)
    all_x -= np.mean(all_x, axis=0)  # normalize
    all_x /= np.std(all_x, axis=0)  # normalize
    all_y = all_x[:, 0].reshape([-1, 1])
    all_x = all_x[:, 1:]
    return all_x, all_y


def load_crappy_formated_csv():
    all_x1 = []
    with open("data/energy_data/consumption.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(15)]
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            if (len(arr) < 16):
                continue
            feature_col = arr[1:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x1.append(np.array(feature_col, dtype=np.float32))

    all_x2 = []
    with open("data/energy_data/consumption2.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(12)]
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            if (len(arr) < 13):
                continue
            feature_col = arr[1:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x2.append(np.array(feature_col, dtype=np.float32))

    all_x3 = []
    with open("data/energy_data/nuclear.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(5)]
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            if (len(arr) < 6):
                continue
            feature_col = arr[1:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x3.append(np.array(feature_col, dtype=np.float32))

    all_x4 = []
    with open("data/energy_data/oil_price.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(6)]
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            if (len(arr) < 7):
                continue
            feature_col = arr[1:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x4.append(np.array(feature_col, dtype=np.float32))

    all_x5 = []
    with open("data/energy_data/renewable.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(13)]
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            if (len(arr) < 14):
                continue
            feature_col = arr[1:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x5.append(np.array(feature_col, dtype=np.float32))

    all_x6 = []
    with open("data/interest_rate_data/matched_data_combined.txt", "r") as f:
        memory = [i for i in range(7)]
        for line in f:
            arr = line.split(",")
            feature_col = arr[:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x6.append(np.array(feature_col, dtype=np.float32))

    assert len(all_x1) == len(all_x2) == len(all_x3) == len(all_x4) == len(all_x5) # == len(all_x6)
    length = len(all_x1)

    all_x = []
    for i in range(length):
        all_x.append(np.concatenate((all_x1[i], all_x2[i], all_x3[i], all_x4[i], all_x5[i], all_x6[i])))

    all_x = np.stack(all_x, axis=0)
    all_x -= np.mean(all_x, axis=0)  # normalize
    all_x /= np.std(all_x, axis=0)  # normalize

    all_y = all_x[:, 14].reshape([-1, 1])
    all_x = np.concatenate((all_x[:,0:14], all_x[:, 15:]), axis = 1)
    return all_x, all_y

def load_pca_combined():
    all_x = []
    with open("data/energy_data/pca_data_7.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(9)]
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            if (len(arr) < 8):
                continue
            feature_col = arr
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x.append(np.array(feature_col, dtype=np.float32))

    all_x = np.stack(all_x, axis=0)

    y_energy = all_x[:, 0].reshape([-1, 1])
    X_energy = all_x[:, 1:]

    print("Energy dataset shape: " + str(X_energy.shape))
    # Load interest rate dataset
    data_path = "data/interest_rate_data/interest_rate_data_all.txt"
    X_ir, y_ir, dates_ir = load_crappy_formated_csv_ir(data_path, 7, False)

    print("Interest rate dataset shape: " + str(X_ir.shape))

    # Get the dates of energy dataset to match
    dates = []
    months = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06",
              "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"}
    with open("data/energy_data/oil_price.txt", "r") as f:
        lineno = -1
        for line in f:
            lineno += 1
            if (lineno == 0):
                continue
            arr = line.split(",")
            date_unchanged = arr[0]
            date_changed = date_unchanged[0:4] + "-" + months[date_unchanged[5:]]
            dates.append(date_changed)

    # Combine the datasets by lining up correct dates
    X_both = np.zeros((X_energy.shape[0], X_energy.shape[1] + X_ir.shape[1]))
    y_both_ir = np.zeros((X_energy.shape[0],))
    for i in range(len(dates)-1):
        X_both[i, :X_energy.shape[1]] = X_energy[i, :]
        date_found = False
        j = 0
        while not date_found and j <= X_ir.shape[0]:
            if dates_ir[j] == dates[i]:
                X_both[i, X_energy.shape[1]:] = X_ir[i, :]
                y_both_ir[i] = y_ir[i]
                date_found = True
            j += 1

    y_both = y_energy
    # y_both = y_both_ir.reshape([-1, 1])
    return X_both, y_both

def cut_in_sequences(x, y, seq_len, inc=1):
    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class CombinedEconomicData:

    def __init__(self, seq_len=16):
        x, y = load_pca_combined()
        print("x both shape: " + str(x.shape))
        print("y both shape: " + str(y.shape))

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


class CombinedEconomicModel:

    def __init__(self, model_type, model_size, learning_rate=0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 14])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

        self.model_size = model_size
        head = self.x
        if (model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type.startswith("ltc")):
            # Was 0.01
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

        self.result_file = os.path.join("results", "../../electricity", "{}_{}.csv".format(model_type, model_size))
        if (not os.path.exists("results/electricity")):
            os.makedirs("results/electricity")
        if (not os.path.isfile(self.result_file)):
            with open(self.result_file, "w") as f:
                f.write("best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n")

        self.checkpoint_path = os.path.join("tf_sessions", "../../electricity", "{}".format(model_type))
        if (not os.path.exists("tf_sessions/electricity")):
            os.makedirs("tf_sessions/electricity")

        self.saver = tf.train.Saver()
        self.loss_tracked = []
        self.preds = []

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, electricity_data, epochs, verbose=True, log_period=50):

        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        for e in range(epochs):
            if (verbose and e % log_period == 0):
                print("self.target_y:    ", str(self.target_y.shape))
                print("energy_data.test_y ", str(electricity_data.test_y.shape))
                test_acc, test_loss, pred = self.sess.run([self.accuracy, self.loss, self.y],
                                                    {self.x: electricity_data.test_x, self.target_y: electricity_data.test_y})
                valid_acc, valid_loss = self.sess.run([self.accuracy, self.loss],
                                                      {self.x: electricity_data.valid_x, self.target_y: electricity_data.valid_y})
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
            for batch_x, batch_y in electricity_data.iterate_train(batch_size=16):
                acc, loss, _ = self.sess.run([self.accuracy, self.loss, self.train_step],
                                             {self.x: batch_x, self.target_y: batch_y})
                if (not self.constrain_op is None):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            self.loss_tracked.append(losses)
            if (verbose and e % log_period == 0):
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.5f}, valid mae: {:0.2f}, test loss: {:0.5f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses), np.mean(accs),
                        valid_loss, valid_acc,
                        test_loss, test_acc
                    ))
            if (e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = best_valid_stats
        print(
            "Best epoch {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.5f}, valid mae: {:0.2f}, test loss: {:0.5f}, test mae: {:0.2f}".format(
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

    linear_loss = np.loadtxt('linear_loss_comb.txt')
    ctrnn_loss = np.loadtxt('ctrnn_loss_comb.txt')
    lstm_loss = np.loadtxt('lstm_loss_comb.txt')
    ltc_loss = np.loadtxt('ltc_loss_comb.txt')

    plt.plot(linear_loss,label="Linear regression")
    plt.plot(ctrnn_loss, label = "CTRNN")
    plt.plot(lstm_loss, label="LSTM")
    plt.plot(ltc_loss, label="LTC")
    plt.yscale('log')
    plt.legend()
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ltc')
    parser.add_argument('--log', default=1, type=int)
    parser.add_argument('--size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    args = parser.parse_args()

    economic_data = CombinedEconomicData()

    tf.reset_default_graph()
    model = CombinedEconomicModel(model_type=args.model, model_size=args.size)
    model.fit(economic_data, epochs=args.epochs, log_period=args.log)
    np.savetxt('ltc_loss_comb.txt', model.loss_tracked)
    plot_losses()


