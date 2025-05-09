from pythonosc import dispatcher
from pythonosc import osc_server
import threading
import time
import csv
import os

# Global variables for the current stage and participant ID.
current_stage = "undefined"
participant_id = "undefined"

def eeg_handler(address, *args):
    timestamp = time.time()
    # Construct a file name using both participant_id and current_stage.
    filename = f"eeg_data_{participant_id}_{current_stage}.csv"
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = ["timestamp"] + [f"data_{i}" for i in range(len(args))]
            writer.writerow(header)
        writer.writerow([timestamp] + list(args))
    print(f"Received OSC message on {address} with values: {args} and written to {filename}")

def start_osc_server(ip="0.0.0.0", port=5000):
    disp = dispatcher.Dispatcher()
    disp.map("/muse/eeg", eeg_handler)
    
    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    print(f"OSC server is listening on {server.server_address}")
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    return server

if __name__ == "__main__":
    server = start_osc_server(ip="0.0.0.0", port=5000)
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting OSC server.")
