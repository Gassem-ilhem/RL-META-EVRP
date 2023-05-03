import io
import json
from os.path import exists

def load_instance(json_file):
    if exists(json_file):
        with io.open(json_file, 'rt', encoding='utf-8', newline='') as file_object:
            instance_data = json.load(file_object)
        instance_data["customers"] = 100
        instance_data["vehicle_battery"] = 1000
        instance_data["vehicle_chargingRate"] = 100
        instance_data["vehicle_dischargingRat"] = 20
        instance_data["vehicle_initialCharging"] = 20
        instance_data["vehicle_avgTravelSpeed"] = 60
        instance_data["customer_charging_time"] = 60
        instance_data["customer_service_time"] = 10
        instance_data["depot_charging_time"] = 60
        instance_data["depot_service_time"] = 10
        return instance_data
    else:
        raise ValueError(f"{json_file} does not exist.")