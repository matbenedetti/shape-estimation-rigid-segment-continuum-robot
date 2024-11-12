#include "Servo.h"

Servo servo;
int const MIN_SERVO = 900;
int const MAX_SERVO = 2100;
int const NULL_POS_MS = 1530;
int const NULL_POS = 96;
int const STEP_SIZE = 0.5;
int const N_MEASUREMENTS = 15;

int max_angle = 70;
int start = -35;
int finalpos = start+max_angle;



bool finished = false;
bool label = false;
char command = 'S';

void setup(void)
{
  Serial.begin(9600);
  servo.attach(9, MIN_SERVO, MAX_SERVO);
  servo.writeMicroseconds(NULL_POS_MS);

  /* Wait for serial on USB platforms. */
  while (!Serial) {
      delay(10);
  }
}

int mapToMs(float deg, int null_pos_ms){
  int ms = map(deg, -50, 50, 1000, 2000);
  return ms + null_pos_ms - 1500;
}



int n = 1;
void loop(void) {
  while (label){
    Serial.println("x,y,z,pos");
    label = false;
  }
  if (n == 4){
    finished = true;
    Serial.println("XXX");
    n += 1;
  }
  if (n < 4){
    Serial.println(n);
    n += 1;
  }
  if (!finished) {
    for (float pos = start; pos <= finalpos; pos += 1){
      if (command == 'S'){
        servo.writeMicroseconds(mapToMs(pos, NULL_POS_MS));
        delay(2);        
      }
      Serial.println(pos);
      // if (sensor.readData(&x, &y, &z)) {
      //   Serial.print(x, 4);
      //   Serial.print(",");
      //   Serial.print(y, 4);
      //   Serial.print(",");
      //   Serial.print(z, 4);
      //   Serial.print(",");
      //   float angle = pos-NULL_POS;
      //   Serial.println(angle);
      // }
    }
    for (float pos = finalpos; pos >= start; pos -= 1){
      if (command == 'S'){
        servo.writeMicroseconds(mapToMs(pos, NULL_POS_MS));
        delay(2);
      }
    }
    Serial.println("finished");
    Serial.println("wait...");
    delay(2);
  }
}
  
