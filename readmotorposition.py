import smaract.scu as scu
import time
import logging

# Set up logging
0
def initialize_devices():
    try:
        scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)
        logging.info("Devices initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize devices: {e}")
        raise

def calibrate_sensor(device_index, channel_index, timeout=30):
    try:
        # Start calibration routine
        result = scu.CalibrateSensor_S(device_index, channel_index)
        logging.info(f"Calibration started for device {device_index}, channel {channel_index}. Result: {result}")
        
        # Set a timeout for calibration to avoid an infinite loop
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = scu.GetStatus_S(device_index, channel_index)
            if status == scu.StatusCode.STOPPED:
                logging.info(f"Calibration complete for device {device_index}, channel {channel_index}.")
                return True
            time.sleep(0.1)  # Small delay to avoid busy-waiting
        
        logging.warning(f"Calibration timed out after {timeout} seconds for device {device_index}, channel {channel_index}.")
        return False
    except Exception as e:
        logging.error(f"Error during calibration: {e}")
        return False

def read_motor_position(device_index, channel_index, prev_position=None):
    try:
        # Call the GetPosition_S function
        position = scu.GetPosition_S(device_index, channel_index)
        
        if position is None:
            logging.error(f"Error reading position for device {device_index}, channel {channel_index}.")
            return prev_position  # Return the previous position if error
        else:
            # Calculate step difference
            if prev_position is not None:
                step = position - prev_position
            else:
                step = 0  # No step if this is the first reading
            
            logging.info(f"Motor position for device {device_index}, channel {channel_index}: {position}, Step: {step}")
            return position  # Return the current position
    except Exception as e:
        logging.error(f"Exception occurred while reading motor position: {e}")
        return prev_position

def main():
    initialize_devices()

    device_index = 0
    try:
        channel_index = int(input("Enter channel index: "))
    except ValueError:
        logging.error("Invalid channel index entered.")
        return

    try:
        calibrate_sensor(device_index, 0)
        calibrate_sensor(device_index, 1)
        calibrate_sensor(device_index, 2)

        prev_position = None  # To store the previous motor position
        
        while True:
            input("Press Enter to read the motor position or Ctrl+C to exit...")
            prev_position = read_motor_position(device_index, channel_index, prev_position)
    except KeyboardInterrupt:
        logging.info("User interrupted the program.")
    finally:
        scu.ReleaseDevices()
        logging.info("Devices released.")

if __name__ == "__main__":
    main()
