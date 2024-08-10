import csv
from keras.callbacks import CSVLogger


# Customizes the CSVLogger utility to include any additional information into the CSV file.
class CustomCSVLogger(CSVLogger):
    def __init__(self, filename, record_name=None, record_data=None, separator=',', append=False):
        super().__init__(filename, separator=separator, append=append)
        self.record_name = record_name
        self.record_data = record_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.record_data:
            logs.update(self.record_data)
        super().on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self.record_data:
            with open(self.filename, 'a') as f:
                writer = csv.writer(f, delimiter=self.sep)
                writer.writerow([self.record_name])
