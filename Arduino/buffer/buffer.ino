#include <Wire.h>

const int sensorAddress = 0x68; // Replace with your sensor's I2C address
unsigned long previousMillis = 0;
const unsigned long interval = 10; // 100 Hz -> 1000 ms / 100 = 10 ms

void setup() {
  Wire.begin(); // Initialize I2C communication
  Serial.begin(9600); // Initialize serial communication for debugging
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    
    Wire.requestFrom(sensorAddress, 1);
    if (Wire.available()) {
      int data = Wire.read();
      Serial.println(data); // Print the data to the serial monitor
    }
  }
}
