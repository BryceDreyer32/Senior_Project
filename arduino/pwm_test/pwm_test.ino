const float PERIOD  = 20;
const int PWM_OUT   = 9;
float   high_time = 0.05;
int   newdc       = 0;

void setup() {
  Serial.begin(115200);
  pinMode(PWM_OUT, OUTPUT);
  Serial.println("Starting PWM Test");
}

void loop() {
  if(Serial.available()) {
    newdc = Serial.parseInt();
    if((newdc > 0 ) & (newdc <= 100)) {
      high_time = (0.0245 * float(newdc)) + 0.05;
      Serial.print("New high time: ");
      Serial.println(high_time);
    }
  }
  digitalWrite(PWM_OUT, HIGH);
  delayMicroseconds(high_time * 1000); 
  digitalWrite(PWM_OUT, LOW);
  delayMicroseconds((PERIOD - high_time) * 1000);
}
