import serial
import time
import numpy as np
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe

no_meas_per_frame = 250
no_bytes_per_meas = 12
fsample = 500

# Configure the serial connection
SERIAL_PORT = "COM3"  # Replace with your actual serial port
BAUD_RATE = 9600  # Must match the baud rate set in the Arduino code

# Open the serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10)

def send_command(command):
    """Send a command to the Arduino via serial."""
    ser.write(command.encode())  # Encode the command to bytes and send it
    time.sleep(0.1)  # Short delay to ensure command is sent

def calculate_mean(data_list):
    """Calculate the mean of a list of values."""
    return np.mean(data_list)

def save_means_to_file(filename, bx_means, by_means, bz_means):
    """Save the mean values to a file."""
    with open(filename, 'w') as file:
        file.write("Mean Bx, Mean By, Mean Bz\n")
        for bx, by, bz in zip(bx_means, by_means, bz_means):
            file.write(f"{bx}, {by}, {bz}\n")

def main():
    # Number of cycles and iterations per cycle
    num_cycles = 3
    iterations_per_cycle = 142

    # Lists to store mean values
    bx_means = []
    by_means = []
    bz_means = []

    try:
        for cycle in range(1, num_cycles + 1):
            for _ in range(iterations_per_cycle):
                # Send the 'S' command
                send_command('S')
                print("Sent 'S' command")

                # Read data from serial port
                databytes = w2bw_read_n_bytes(no_meas_per_frame * no_bytes_per_meas, "COM8", fsample)

                # Process the data
                data = read_w2bw_tms320_data_syncframe(databytes)

                Bxs = data["Bxs"]
                Bys = data["Bys"]
                Bzs = data["Bzs"]

                # Calculate means
                mean_bx = calculate_mean(Bxs)
                mean_by = calculate_mean(Bys)
                mean_bz = calculate_mean(Bzs)

                # Store the means
                bx_means.append(mean_bx)
                by_means.append(mean_by)
                bz_means.append(mean_bz)

                # Print means for debugging
                print(f"Mean Bx: {mean_bx}")
                print(f"Mean By: {mean_by}")
                print(f"Mean Bz: {mean_bz}")

                time.sleep(1)  # Adjust the delay as needed

            # Save the means to a file
            filename = f'mean_values{cycle}.txt'
            save_means_to_file(filename, bx_means, by_means, bz_means)

            # Clear the lists for the next cycle
            bx_means.clear()
            by_means.clear()
            bz_means.clear()

    except KeyboardInterrupt:
        print("Program interrupted. Closing...")
    finally:
        if ser.is_open:
           ser.close()  # Close the serial connection when done
        pass

if __name__ == '__main__':
    main()
