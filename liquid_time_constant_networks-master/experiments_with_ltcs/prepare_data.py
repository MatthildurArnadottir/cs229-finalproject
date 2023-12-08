import numpy as np
import matplotlib.pyplot as plt

def sort_values(dates, values):
    combined = []
    for i in range(len(dates)):
        sample = [dates[i], values[i]]
        combined.append(sample)
    if combined[0][0][0:4] == '2023':
        combined.reverse()
    return combined


def load_feature(csv_path, value_label):
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
    dates_cols = [i for i in range(len(headers)) if headers[i].startswith('DATE')]
    values_cols = [i for i in range(len(headers)) if headers[i].startswith(value_label)]
    dates = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=dates_cols, dtype=str)
    values = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=values_cols)
    data = sort_values(dates, values)
    return data

def combine_features(csv_paths,value_labels):
    cpi_data = load_feature(csv_paths[0],value_labels[0])
    rate_data = load_feature(csv_paths[1], value_labels[1])
    gdp_data = load_feature(csv_paths[2], value_labels[2])
    sp500_data = load_feature(csv_paths[3], value_labels[3])
    tb1_data = load_feature(csv_paths[4], value_labels[4])
    tb10_data = load_feature(csv_paths[5], value_labels[5])
    unemployment_data = load_feature(csv_paths[6], value_labels[6])
    debt_data = load_feature(csv_paths[7], value_labels[7])

    combined = []
    # Add dates and interest rates (len=2)
    for i in range(len(rate_data)):
        combined.append(rate_data[i])
    # Add CPI and GDP (len=4)
    for i in range(len(combined)):
        for j in range(len(cpi_data)):
            if cpi_data[j][0] == combined[i][0]:
                combined[i].append(cpi_data[j][1])
        for j in range(len(gdp_data)):
            if gdp_data[j][0] == combined[i][0]:
                combined[i].append(gdp_data[j][1])
    # If data is missing, add previous entry
    for i in range(len(combined)):
        if len(combined[i]) == 3:
            combined[i].append(combined[i-1][3])
    # Add S&P 500 data (len=5)
    for i in range(len(combined)):
        for j in range(len(sp500_data)):
            if sp500_data[j][0] == combined[i][0]:
                combined[i].append(sp500_data[j][1])
    for i in range(len(combined)):
        if len(combined[i]) == 4:
            combined[i].append("")
    # Add Treasury Bond data (len=7)
    for i in range(len(combined)):
        for j in range(len(tb1_data)):
            if tb1_data[j][0] == combined[i][0]:
                combined[i].append(tb1_data[j][1])
    for i in range(len(combined)):
        if len(combined[i]) == 5:
            combined[i].append("")
    for i in range(len(combined)):
        for j in range(len(tb10_data)):
            if tb10_data[j][0] == combined[i][0]:
                combined[i].append(tb10_data[j][1])
    for i in range(len(combined)):
        if len(combined[i]) == 6:
            combined[i].append("")
    # Add unemployment data (len=8)
    for i in range(len(combined)):
        for j in range(len(unemployment_data)):
            if unemployment_data[j][0] == combined[i][0]:
                combined[i].append(unemployment_data[j][1])
    for i in range(len(combined)):
        if len(combined[i]) == 7:
            combined[i].append("")
    # Add debt data (len=9)
    for i in range(len(combined)):
        for j in range(len(debt_data)):
            if debt_data[j][0] == combined[i][0]:
                combined[i].append(debt_data[j][1])
    for i in range(len(combined)):
        if len(combined[i]) == 7:
            combined[i].append("")

    header_labels = ['Date',
                     'Interest_rate',
                     'CPI',
                     'GDP',
                     'S&P500',
                     'Treasury_Bond_1year',
                     'Treasury_Bond_10year',
                     'Unemployment',
                     'Debt'
                     ]
    combined.insert(0, header_labels)
    np.savetxt(r"data/interest_rate_data/interest_rate_data_all.txt", combined, fmt="%s", delimiter=';')

    return combined


def run_data_preparation():

    csv_paths = [
        'data/interest_rate_data/cpi.csv',
        'data/interest_rate_data/effective_rate.csv',
        'data/interest_rate_data/GDP.csv',
        'data/interest_rate_data/s&p500.csv',
        'data/interest_rate_data/treasury_bond_1_year.csv',
        'data/interest_rate_data/treasury_bond_10_year.csv',
        'data/interest_rate_data/unemployment.csv',
        'data/interest_rate_data/us_debt.csv'
    ]
    value_labels = [
        'CPIAUCSL', 'FEDFUNDS', 'GDP', 'Value', 'GS1', 'GS10', 'UNRATE', 'DEBT'
    ]

    data = combine_features(csv_paths, value_labels)

    data_matrix = np.zeros((len(data),len(data[0])))
    dates = []
    for i in range(1,len(data)):
        for j in range(7):
            data_matrix[i,j] = data[i][j+1]
        dates.append(data[i][0])

    plt.figure(figsize=(10, 6))
    plt.plot(data_matrix[1:,0])
    plt.show()


if __name__ == "__main__":
    run_data_preparation()
