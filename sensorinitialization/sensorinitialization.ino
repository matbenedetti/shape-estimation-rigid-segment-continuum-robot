#include <Wire.h>

#define AK09973D_I2C_ADDR 0x11 // Correct I2C address for AK09973D

void setup() {
  Serial.begin(9600); // Use a more common baud rate for debugging
  Wire.begin();

  // Initialize the AK09973D
  if (!initAK09973D()) {
    Serial.println("No sensor found ... check your wiring?");
    while (1) {
      delay(10);
    }
  }
  Serial.println("AK09973D sensor initialized");
}

void loop() {

}

bool initAK09973D() {
  Wire.beginTransmission(AK09973D_I2C_ADDR);
  Wire.write(0x21);
  Wire.write(0x02);
  if (Wire.endTransmission() != 0) {
    Serial.println("Failed to initialize AK09973D");
    return false;
  }
  Serial.println("AK09973D initialization successful");
  return true;
}

