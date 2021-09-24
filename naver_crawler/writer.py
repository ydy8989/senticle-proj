import csv
import platform
import os
from naver_crawler.exceptions import *

class Writer(object):
    def __init__(self, query, ds, de):
        self.file = None
        self.ds = ds
        self.de = de
        self.initialize_file(query, self.ds, self.de)

        self.csv_writer = csv.writer(self.file)

    def initialize_file(self, query, ds, de):
        output_path = f'../data'
        if os.path.exists(output_path) is not True:
            os.mkdir(output_path)

        file_name = f'{output_path}/{query}_{ds}_{de}.csv'
        if os.path.isfile(file_name):
            raise ExistFile(file_name)

        user_os = str(platform.system())
        if user_os == "Windows":
            self.file = open(file_name, 'w', encoding='euc-kr', newline='')
        # Other OS uses utf-8
        else:
            self.file = open(file_name, 'w', encoding='utf-8', newline='')

    def write_row(self, arg):
        self.csv_writer.writerow(arg)

    def close(self):
        self.file.close()