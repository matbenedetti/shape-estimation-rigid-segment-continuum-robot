#include "Servo.h"


Servo servo;
int const MIN_SERVO = 900;
int const MAX_SERVO = 2100;
int const NULL_POS_MS = 1530;
int const NULL_POS = 96;
float prevangle = 0;


void setup(void)
{
  Serial.begin(4800);
  Serial.flush();
  servo.attach(9, MIN_SERVO, MAX_SERVO);
  servo.writeMicroseconds(1530);

  /* Wait for serial on USB platforms. */
  while (!Serial) {
      delay(10);
  }
}


int mapToMs(float deg, int null_pos_ms){
  int ms = map(deg, -50, 50, 1000, 2000);
  return ms + null_pos_ms - 1500;
}


void loop(void) {
  while (!Serial.available());
  float angle = Serial.readString().toFloat();
  if (angle == 0 && prevangle != 0) {
    angle = prevangle;
  } else {
    prevangle = angle;
  }
  servo.writeMicroseconds(mapToMs(angle, NULL_POS_MS));
  Serial.println(angle);
  delay(2);
  Serial.println("FM");
}
