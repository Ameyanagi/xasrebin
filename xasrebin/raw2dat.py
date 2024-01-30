import numpy as np
import pandas as pd
from glob import glob

# import bin
import os

valve_possitions = {
    0: "H2",
    1: "O2",
}


def read_file_to_df(file, columns=None, return_columns=None, valve_tol=2.5):
    if columns is None:
        columns = [
            "timestamp",
            "i0",
            "it",
            "ir",
            "iff",
            "aux1",
            "aux2",
            "aux3",
            "aux4",
            "energy",
        ]

    if return_columns is None:
        return_columns = ["timestamp", "i0", "it", "ir", "iff", "aux2", "energy"]

    data = np.loadtxt(file)
    df = pd.DataFrame(data, columns=columns)
    df = df[return_columns]
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(inplace=True, drop=True)

    df["aux2"] = df["aux2"].apply(lambda x: 1 if x > valve_tol else 0)

    return df, find_valve_changes(df)


def read_files_to_df(files, columns=None, return_columns=None, valve_tol=2.5):
    for file in files:
        df, vc = read_file_to_df(file, columns, return_columns, valve_tol)

        if file == files[0]:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    return df_all


def find_valve_changes(df):
    valve_changes = df["aux2"].diff().abs()

    valve_changes = valve_changes[valve_changes > 0]
    valve_changes = df.iloc[valve_changes.index]["timestamp"].values

    return valve_changes


def find_valve_changes_timestamps(files):
    valve_changes = np.array([])

    for file in files:
        df, vc = read_file_to_df(file)

        valve_changes = np.append(valve_changes, vc)

    valve_changes = np.sort(valve_changes)
    valve_changes_diff = np.diff(valve_changes)

    valve_changes_mean = np.mean(valve_changes_diff)

    valve_changes_mean = valve_changes_diff[
        valve_changes_diff < valve_changes_mean * 1.5
    ].mean()

    valve_change_times = np.array([])

    for i, change in enumerate(valve_changes_diff):
        valve_change_number = int(np.round(change / valve_changes_mean / 1.2))

        for j in range(valve_change_number):
            valve_change_times = np.append(
                valve_change_times,
                valve_changes[i + 1] - change / valve_change_number * j,
            )

    valve_change_times = np.sort(valve_change_times)

    return valve_change_times


def find_time_diff(timestamp, valve_changes_array):
    nearest_valve_change = np.searchsorted(valve_changes_array, timestamp)

    print(len(valve_changes_array))

    return timestamp - valve_changes_array[nearest_valve_change - 1]


def xas_energy_grid(
    energy_range,
    e0=24350,
    edge_start=-30,
    edge_end=50,
    preedge_spacing=5,
    xanes_spacing=0.5,
    exafs_k_spacing=0.05,
):
    energy_range_lo = np.min(energy_range)
    energy_range_hi = np.max(energy_range)

    preedge = np.arange(energy_range_lo, e0 + edge_start - 1, preedge_spacing)

    before_edge = np.arange(e0 + edge_start, e0 + edge_start + 7, 1)

    edge = np.arange(e0 + edge_start + 7, e0 + edge_end - 7, xanes_spacing)

    after_edge = np.arange(e0 + edge_end - 7, e0 + edge_end, 0.7)

    eenergy = k2e(e2k(e0 + edge_end, e0), e0)
    post_edge = np.array([])

    while eenergy < energy_range_hi:
        kenergy = e2k(eenergy, e0)
        kenergy += exafs_k_spacing
        eenergy = k2e(kenergy, e0)
        post_edge = np.append(post_edge, eenergy)
    return np.concatenate((preedge, before_edge, edge, after_edge, post_edge))


def k2e(k, E0):
    """
    Convert from k-space to energy in eV

    Parameters
    ----------
    k : float
        k value
    E0 : float
        Edge energy in eV

    Returns
    -------
    out : float
        Energy value

    See Also
    --------
    :func:`isstools.conversions.xray.e2k`
    """
    return ((1000 / (16.2009**2)) * (k**2)) + E0


def e2k(E, E0):
    """
    Convert from energy in eV to k-space

    Parameters
    ----------
    E : float
        Current energy in eV
    E0 : float
        Edge energy in eV

    Returns
    -------
    out : float
        k-space value

    See Also
    --------
    :func:`isstools.conversions.xray.k2e`
    """
    return 16.2009 * (((E - E0) / 1000) ** 0.5)


def calc_async_spectra(
    df, bin_size, time_min=0, time_max=15, e0=24350, output_dir="data"
):
    valve_unique = df["aux2"].unique()

    time_range = np.linspace(time_min, time_max, bin_size + 1)

    time_range_list = []

    for i in range(len(time_range) - 1):
        time_range_list.append([time_range[i], time_range[i + 1]])

    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    for valve in valve_unique:
        df_valve = df[df["aux2"] == valve]

        for time_range in time_range_list:
            df_time = df_valve[
                (df_valve["delay"].between(time_range[0], time_range[1]))
            ]

            # do convolution
            df_result = df.sort_values(by="energy").reset_index(drop=True)

            energy_grid = xas_energy_grid(df_result["energy"].values, e0)

            colomns = df_result.columns

            df_rebinned = rebin(df_time, energy_grid)

            df_rebinned.drop(columns=["timestamp", "delay"], inplace=True)
            df_rebinned.to_csv(
                output_dir
                + "/{}_start{}_end{}.csv".format(
                    valve_possitions[valve], time_range[0], time_range[1]
                ),
                index=False,
            )


def rebin(df, energy_grid):
    for i in range(len(energy_grid) - 1):
        data = df[df["energy"].between(energy_grid[i], energy_grid[i + 1])].mean()
        data = data.to_frame().T
        if i == 0:
            df_rebinned = data
        df_rebinned = pd.concat([df_rebinned, data], axis=0)

    return df_rebinned


def load_df(file, columns=None):
    if columns is None:
        columns = [
            "timestamp",
            "i0",
            "it",
            "ir",
            "iff",
            "aux1",
            "aux2",
            "aux3",
            "aux4",
            "energy",
        ]

    data = np.loadtxt(file)
    df = pd.DataFrame(data, columns=columns)
    df.sort_values(by="timestamp", inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def generate_files_to_raw_df(files, save_path="output.csv"):
    files = glob(files)
    print(files)
    valve_changes_times = find_valve_changes_timestamps(files)

    df = read_files_to_df(files)

    df["delay"] = df["timestamp"].apply(
        lambda x: find_time_diff(x, valve_changes_times)
    )

    print(df.head())
    df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    # df = load_df("raw_data_Pd8/Pd8_S1b_over_#11_10_O2_10_H2_at_250 at X 6.564 Y -6.846 Pd-30s_scans_async 0001 0001.raw")
    # df = rebin(df, xas_energy_grid(df["energy"].values, e0=24350))

    # df.to_csv("test.csv", index=False)

    # data = np.loadtxt("Pd8_S1b_over_#11_10_O2_10_H2_at_250 at X 6.564 Y -6.846 Pd-30s_scans_async 0001 0001.dat")

    # import matplotlib.pyplot as plt

    # plt.plot(data[:, 0], data[:, 4]/data[:, 1], label="Convolution(beamline)")
    # plt.plot(df["energy"], df["iff"]/df["i0"], label="Convolution(my method)")

    # plt.legend()
    # plt.show()

    files = glob("N1 Co 3Fe Filter  0007.raw")
    # valve_changes_times = find_valve_changes_timestamps(files)

    df = read_files_to_df(files)

    energy_grid = xas_energy_grid(df["energy"].values, e0=7709)

    df = rebin(df, energy_grid)

    df.to_csv("N1.dat", index=False)

    import matplotlib.pyplot as plt

    plt.plot(df["energy"], df["iff"] / df["i0"])
    plt.plot(df["energy"], df["i0"])
    plt.show()

    # # df["delay"] = df["timestamp"].apply(lambda x: find_time_diff(x, valve_changes_times))

    # # df.to_csv("Pd25-13.csv", index=False)
    # save_path = "./Pd25-13.csv"
    # save_dir = "./data-Pd25-13/"
    # df = generate_files_to_raw_df("raw_data/*.raw", save_path=save_path)

    # # df = pd.read_csv("Pd8-11.csv")
    # calc_async_spectra(df, 6, time_max=10, output_dir="data/")

    # save_path = "./Pd8-09.csv"
    # save_dir = "./data-Pd8-09/"
    # df = generate_files_to_raw_df("raw_data_Pd8/*.raw", save_path=save_path)

    # df = pd.read_csv("Pd8-11.csv")
    # calc_async_spectra(df, 10, time_max=10, output_dir="data/")
