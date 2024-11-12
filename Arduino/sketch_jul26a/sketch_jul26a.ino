int signalPin1 = A0; // Pin where the first signal is connected
int signalPin2 = A1; // Pin where the second signal is connected

void setup() {
  Serial.begin(2000000);
}

void loop() {
  int readValue1 = analogRead(signalPin1);
  int readValue2 = analogRead(signalPin2);

  float voltage1 = readValue1 * (5.0 / 1023.0) * 1000; // Convert ADC value to mV
  float voltage2 = readValue2 * (5.0 / 1023.0) * 1000; // Convert ADC value to mV
  float drop = voltage1 - voltage2; // Voltage drop in mV

  // Calculate current assuming the shunt resistor is 10 ohms
  float current = drop / 10.0; // Current in mA

  // Print current value for Serial Plotter
  Serial.println(voltage2);

}