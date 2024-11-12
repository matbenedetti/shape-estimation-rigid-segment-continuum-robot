#include <Servo.h>

Servo servo;
const int MIN_SERVO = 900;
const int MAX_SERVO = 2100;
const int NULL_POS_MS = 1530;
const int NULL_POS = 96;

void setup() {
  Serial.begin(115200);
  Serial.flush();
  servo.attach(9, MIN_SERVO, MAX_SERVO);
  servo.writeMicroseconds(NULL_POS_MS);

  // Wait for serial on USB platforms
  while (!Serial) {
    delay(10);
  }
}

int mapToMs(float deg, int null_pos_ms) {
  int ms = map(deg, -50, 50, MIN_SERVO, 2000);
  return ms + null_pos_ms - 1500;
}
void loop(void) {
  while (!Serial.available());
  float angle = Serial.readString().toFloat();
  servo.writeMicroseconds(mapToMs(angle, NULL_POS_MS));
  delay(2);
  Serial.print("FM");
}

