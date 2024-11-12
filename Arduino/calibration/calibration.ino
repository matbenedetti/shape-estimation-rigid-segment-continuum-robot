#include <Wire.h>

#define AK09973D_I2C_ADDR 0x010  // The I2C address for AK09973D (forse fai un check)

void setup() {
  Serial.begin(2000000);
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
  if (Serial.available()) {
    int receivedValue = Serial.parseInt();
    if (receivedValue == 123) {
      float x, y, z;
      if (readMagneticData(x, y, z)) {
        Serial.print("PPP;");
        Serial.print(x, 4);
        Serial.print(";");
        Serial.print(y, 4);
        Serial.print(";");
        Serial.println(z, 4);
      } else {
        Serial.println("Failed to read data from sensor");
      }
    }
  }
}

bool initAK09973D() {
  Wire.beginTransmission(AK09973D_I2C_ADDR);
  Wire.write(0x30); // Register to read
  Wire.write(0x01); // Value to set
  if (Wire.endTransmission() != 0) {
    return false;
  }
  return true;
}

bool readMagneticData(float &x, float &y, float &z) {
  Wire.beginTransmission(AK09973D_I2C_ADDR);
  Wire.write(0x17); // Register to start reading
  if (Wire.endTransmission() != 0) {
    return false;
  }

  Wire.requestFrom(AK09973D_I2C_ADDR, 7);
  if (Wire.available() < 6) {
    return false;
  }

  uint8_t ST = Wire.read();
  uint8_t zh = Wire.read();
  uint8_t zl = Wire.read();
  uint8_t yh = Wire.read();
  uint8_t yl = Wire.read();
  uint8_t xh = Wire.read();
  uint8_t xl = Wire.read();

  x = (int16_t)(xh << 8 | xl) * 0.15; // da controllare
  y = (int16_t)(yh << 8 | yl);
  z = (int16_t)(zh << 8 | zl);

  return true;
}
