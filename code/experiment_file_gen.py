import glob
import numpy as np
from collections import Counter
import os.path
import multiprocessing
import time
import pickle

class ListMaintenance:
    @staticmethod
    def get_index_from_file_name(file):
        index1 = file.rfind('-')
        index2 = file.rfind('.')
        return int(file[index1 + 1:index2])

    @staticmethod
    def get_host_from_file_name(file):
        index1 = file.rfind(os.sep)
        index1 = max(index1 + 1, 0)
        index2 = file.find("-")
        return file[index1:index2]

    @staticmethod
    def is_file_valid(file):
        data = np.load(file)["arr_0"]
        counter = Counter(data)
        if counter[1] >= 200 and counter[-1] > 1000:
            return True, file
        else:
            return False, ""

    def __init__(self, path):
        self.path = path
        self.valid_files = []
        self._handled_files = 0
        self.valid_files_path = os.path.join(self.path, "0_valid_files")
        self.sites_list_path = os.path.join(self.path, "0_sites_list")
        self.bgr_files_path = os.path.join(self.path, "0_bgr_files")
        self.sites_list = [[], ]
        self.bgr_file_list = []

    def generate_bgr_files_list(self):
        sites_dict = {}
        raw_file_list = list(glob.glob(os.path.join(self.path, "*.npz")))
        for a_file in raw_file_list:
            a_host = self.get_host_from_file_name(a_file)
            if a_host not in sites_dict:
                sites_dict[a_host] = list()
            index = a_file.rfind(os.sep)
            sites_dict[a_host].append(a_file[index + 1:])
        for a_host, a_host_list in sites_dict.items():
            if len(a_host_list) <= 6:
                self.bgr_file_list.extend(a_host_list)

        with open(self.bgr_files_path, "wb") as f:
            pickle.dump(self.bgr_file_list, f)

    def load_bgr_file_list(self):
        if not os.path.exists(self.bgr_files_path):
            self.generate_bgr_files_list()
        with open(self.bgr_files_path, "rb") as f:
            self.bgr_file_list = pickle.load(f)

    def generate_valid_files_list(self):
        pool = multiprocessing.Pool(processes=30)
        raw_file_list = list(glob.glob(os.path.join(self.path, "*.npz")))
        raw_file_list_len = len(raw_file_list)
        for a_file in raw_file_list:
            pool.apply_async(self.is_file_valid, args=(a_file,), callback=self.generate_valid_files_list_callback)
        pool.close()
        while True:
            print(
                f"Total: {raw_file_list_len}, Handled: {self._handled_files}, {round(self._handled_files / raw_file_list_len * 100, 3)}%")
            if self._handled_files / raw_file_list_len > 0.98:
                break
            time.sleep(20)
        pool.join()
        valid_files_len = len(self.valid_files)
        print(f"Valid files: {valid_files_len}")
        with open(self.valid_files_path, "wb") as f:
            pickle.dump(self.valid_files, f)

    def generate_valid_files_list_callback(self, param):
        self._handled_files += 1
        if param[0]:
            index = param[1].rfind(os.sep) + 1
            self.valid_files.append(param[1][index:])

    def load_valid_files_list(self):
        if not os.path.exists(self.valid_files_path):
            self.generate_valid_files_list()
        with open(self.valid_files_path, "rb") as f:
            self.valid_files = pickle.load(f)

    def generate_sites_list(self):
        if len(self.valid_files) == 0:
            self.load_valid_files_list()
        sites_dict = {}
        for a_file in self.valid_files:
            host = self.get_host_from_file_name(a_file)
            if host not in sites_dict:
                sites_dict[host] = []
            sites_dict[host].append(a_file)
        for a_host, a_host_list in sites_dict.items():
            if len(a_host_list) <= 6:
                self.sites_list[0].extend(a_host_list)
            elif len(a_host_list) >= 500:
                self.sites_list.append(a_host_list)
        with open(self.sites_list_path, "wb") as f:
            pickle.dump(self.sites_list, f)

    def load_sites_list(self):
        if not os.path.exists(self.sites_list_path):
            self.generate_sites_list()
        with open(self.sites_list_path, "rb") as f:
            self.sites_list = pickle.load(f)

    def generate_sites_list_subset(self, number_of_mon, is_open_world=False, is_split=False):
        if len(self.sites_list) == 1:
            self.load_sites_list()
        output_file = f"0_{number_of_mon}_sites_list"
        tmp_sites_list = self.sites_list[1:1 + number_of_mon]
        sites_list = []
        if is_split:
            output_file = output_file + "_split"
            quic = []
            tcp = []
            for a_host_list in tmp_sites_list:
                tmp_i = 0
                tmp_j = 0
                while True:
                    file_index1 = self.get_index_from_file_name(a_host_list[tmp_i])
                    if file_index1 < 275:
                        break
                    tmp_i += 1
                while True:
                    file_index2 = self.get_index_from_file_name(a_host_list[tmp_j])
                    if file_index2 >= 275:
                        break
                    tmp_j += 1

                quic.append(a_host_list[tmp_i])
                tcp.append(a_host_list[tmp_j])
            sites_list.append(quic)
            sites_list.append(tcp)
        else:
            output_file = output_file + "_mix"
            sites_list = tmp_sites_list

        if is_open_world:
            output_file = output_file + "_open"
            tmp_list = []
            for a_host_list in sites_list:
                tmp_list.extend(a_host_list)
            sites_list = [self.sites_list[0], tmp_list]
        else:
            output_file = output_file + "_close"

        with open(os.path.join(self.path, output_file), "wb") as f:
            pickle.dump(sites_list, f)

    def generate_sites_list_subset_quic(self, number):
        output_name = f"0_{number}_quic_close"
        if len(self.sites_list) == 1:
            self.load_sites_list()
        tmp_sites_list = self.sites_list[1:1 + number]
        sites_list = []
        for a_site_list in tmp_sites_list:
            tmp_list = []
            for a_file in a_site_list:
                file_index = self.get_index_from_file_name(a_file)
                if file_index < 275:
                    tmp_list.append(a_file)
            sites_list.append(tmp_list)
        with open(os.path.join(self.path, output_name), "wb") as f:
            pickle.dump(sites_list, f)

    def generate_sites_list_subset_split(self, number):
        output_name = f"0_{number}_split"
        if len(self.sites_list) == 1:
            self.load_sites_list()
        tmp_sites_list = self.sites_list[1:1 + number]
        sites_list = []
        for a_site_list in tmp_sites_list:
            quic = []
            tcp = []
            for a_file in a_site_list:
                file_index = self.get_index_from_file_name(a_file)
                if file_index < 275:
                    quic.append(a_file)
                else:
                    tcp.append(a_file)
            sites_list.append(quic)
            sites_list.append(tcp)
        with open(os.path.join(self.path, output_name), "wb") as f:
            pickle.dump(sites_list, f)

    def generate_sites_list_open_quic(self, number_of_mon, number_of_mon_trac, number_of_unmon):
        output_name = f"0_open_quic_{number_of_mon}_{number_of_mon_trac}_{number_of_unmon}"
        result = [[], []]
        if len(self.sites_list) == 1:
            self.load_sites_list()
        if len(self.bgr_file_list) == 0:
            self.load_bgr_file_list()
        result[0].extend(self.bgr_file_list[:number_of_unmon])
        for a_host_list in self.sites_list[1:1 + number_of_mon]:
            tmp_host_list = a_host_list[:]
            tmp_host_list.sort(key=self.get_index_from_file_name)
            result[1].extend(tmp_host_list[:number_of_mon_trac])
        with open(os.path.join(self.path, output_name), "wb") as f:
            pickle.dump(result, f)

    def generate_sites_list_open_mix(self, number_of_mon, number_of_mon_trace, number_of_total_trace=None):
        if number_of_total_trace is None:
            number_of_total_trace = number_of_mon * number_of_mon_trace
        half_trace = number_of_mon_trace // 2
        output_name = f"0_open_mix_{number_of_mon}_{number_of_mon_trace}_{number_of_total_trace}"
        result = [[], []]
        if len(self.sites_list) == 1:
            self.load_sites_list()
        if len(self.bgr_file_list) == 0:
            self.load_bgr_file_list()
        result[0].extend(self.bgr_file_list[:number_of_total_trace])
        for a_host_list in self.sites_list[1:1 + number_of_mon]:
            tmp_host_list = a_host_list[:]
            tmp_host_list.sort(key=self.get_index_from_file_name)
            result[1].extend(tmp_host_list[:half_trace])
            result[1].extend(tmp_host_list[275:275 + half_trace])
        with open(os.path.join(self.path, output_name), "wb") as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    maintance = ListMaintenance(r"D:\wfp\QUIC-TOR")
    maintance.generate_sites_list_subset_quic(100)