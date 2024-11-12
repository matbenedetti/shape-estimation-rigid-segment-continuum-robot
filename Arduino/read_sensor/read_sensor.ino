#include <Wire.h>
#include <HardwareSerial.h>

// Pin Definitions
const int SENSOR_POWER_PIN = 11;  // Replace with the actual pin used for sensor power control
const int LED_PIN = 4;            // LED pin

// I2C Address
const uint8_t DEVICE_ADDRESS = 0x6A;  // Change if needed

// Register Addresses
const uint8_t ADDR_CONFIG = 0x10;
const uint8_t ADDR_MOD = 0x11;
const uint8_t ADDR_CONFIG2 = 0x14;

// Trigger Modes
const uint8_t TRIGGER_NONE = 0x00;
const uint8_t TRIGGER_AFTER_WRITE_FRAME = 0b00100000;

// Bit-Fields within Registers
const uint8_t CONFIG_REG_MASK_TRIG_AFTER5H = 0b00100000;
const uint8_t CONFIG_REG_ODD_PARITY_BIT = 0b00000001;
const uint8_t CONFIG_REG_MASK_X2_SENS = 0b00001000;
const uint8_t CONFIG_REG_DISABLE_EMP = 0b10000000;

const uint8_t MOD_REG_MASK_MODE_MASTER = 0b00000001;
const uint8_t MOD_REG_MASK_INT_DISABLE = 0b00000100;
const uint8_t MOD_REG_MASK_ONEBYTE_EN = 0b00010000;
const uint8_t MOD_REG_MASK_ODD_PARITY_BIT = 0b10000000;
const uint8_t MOD_REG_MASK_A1 = 0b00100000;

const uint8_t MOD2_REG_MASK_X4_SENS = 0b00000001;

// Configuration Registers
const uint8_t WRITE_CONFIG_CONFIG_REG[] = {ADDR_CONFIG, CONFIG_REG_MASK_TRIG_AFTER5H | CONFIG_REG_MASK_X2_SENS | CONFIG_REG_ODD_PARITY_BIT};
const uint8_t WRITE_CONFIG_MOD1_REG[] = {ADDR_MOD, MOD_REG_MASK_MODE_MASTER | MOD_REG_MASK_ONEBYTE_EN | MOD_REG_MASK_ODD_PARITY_BIT};
const uint8_t WRITE_CONFIG_CONFIG2_REG[] = {ADDR_CONFIG2, MOD2_REG_MASK_X4_SENS};

// Data Structures
struct w2bw_meas {
    int16_t Bx;
    int16_t By;
    int16_t Bz;
    uint16_t Temp;
};

// Function Prototypes
void parse_w2bw_meas(const uint8_t* bytes, struct w2bw_meas* meas);
void serialize_w2bw(uint8_t* dest, const struct w2bw_meas* meas);

// Timing Variables
unsigned long lastTime = 0;
const unsigned long interval = 1000; // Interval for loop in milliseconds

void setup() {
    Serial.begin(460800);
    Wire.begin();

    pinMode(SENSOR_POWER_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);

    // Power cycle the sensor
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(200);
    digitalWrite(SENSOR_POWER_PIN, LOW);
    delay(200);

    // Initialize the sensor
    Wire.beginTransmission(DEVICE_ADDRESS);
    Wire.write(WRITE_CONFIG_CONFIG_REG, sizeof(WRITE_CONFIG_CONFIG_REG));
    Wire.endTransmission();
    delay(10);

    Wire.beginTransmission(DEVICE_ADDRESS);
    Wire.write(WRITE_CONFIG_MOD1_REG, sizeof(WRITE_CONFIG_MOD1_REG));
    Wire.endTransmission();
    delay(10);

    Wire.beginTransmission(DEVICE_ADDRESS);
    Wire.write(WRITE_CONFIG_CONFIG2_REG, sizeof(WRITE_CONFIG_CONFIG2_REG));
    Wire.endTransmission();
    delay(10);
}

void loop() {
    unsigned long currentTime = millis();
    if (currentTime - lastTime >= interval) {
        lastTime = currentTime;

        digitalWrite(LED_PIN, !digitalRead(LED_PIN));

        Wire.requestFrom(DEVICE_ADDRESS, 7);
        uint8_t i2c_data_buffer[7];
        int i = 0;
        while (Wire.available() && i < 7) {
            i2c_data_buffer[i++] = Wire.read();
        }

        struct w2bw_meas current_meas;
        parse_w2bw_meas(i2c_data_buffer, &current_meas);

        uint8_t sendbuf[8];
        serialize_w2bw(sendbuf, &current_meas);

        Serial.write(sendbuf, sizeof(sendbuf));
    }
}

void parse_w2bw_meas(const uint8_t* bytes, struct w2bw_meas* meas) {
    uint16_t Bx, By, Bz, Temp;
    Bx = By = Bz = Temp = 0;

    Bx = bytes[0] << 4;
    if (bytes[0] & 0b10000000) Bx |= 0xF000;
    By = bytes[1] << 4;
    if (bytes[1] & 0b10000000) By |= 0xF000;
    Bz = bytes[2] << 4;
    if (bytes[2] & 0b10000000) Bz |= 0xF000;
    Temp = bytes[3] << 4;
    Bx += (bytes[4] & 0b11110000) >> 4;
    By += (bytes[4] & 0b00001111);
    Bz += (bytes[5] & 0b00001111);
    Temp += (bytes[5] & 0b11000000) >> 6;

    meas->Bx = (int16_t) Bx;
    meas->By = (int16_t) By;
    meas->Bz = (int16_t) Bz;
    meas->Temp = Temp;
}

void serialize_w2bw(uint8_t* dest, const struct w2bw_meas* meas) {
    dest[0] = 0x55;  // Frame start
    dest[1] = 0x55;  // Frame start
    dest[2] = (meas->Bx);
    dest[3] = (meas->Bx) >> 8;
    dest[4] = (meas->By);
    dest[5] = (meas->By) >> 8;
    dest[6] = (meas->Bz);
    dest[7] = (meas->Bz) >> 8;
}
