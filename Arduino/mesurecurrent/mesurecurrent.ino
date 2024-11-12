int signalPin1 = A0; // Pin where the first signal is connected

void setup() {
  Serial.begin(2000000); // Increased baud rate for faster data transmission
}

void loop() {
  float readValue1 = analogRead(signalPin1);

  float voltage1 = ((readValue1 * (5.0 / 1023.0))* 1000); // Convert ADC value to mV


  // Print voltage1 for Serial Plotter
  Serial.println(voltage1);
}
