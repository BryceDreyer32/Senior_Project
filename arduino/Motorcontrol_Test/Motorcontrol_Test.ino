int BRAKE_PIN = 0;
int PWM_PIN = 1;
int ENABLE_PIN = 2;
int DIRECTION_PIN = 3;
int FAULT_PIN = 4;
int ADC_PIN = 5;
int GND_PIN = 6;
int PWR_PIN = 7;

int pwm_period = 14;
int high_time = pwm_period * 0.2;

void setup() {
  // Set port directions
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIRECTION_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);
  pinMode(FAULT_PIN, INPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(GND_PIN, OUTPUT);

  // Initial pin values
  digitalWrite(ENABLE_PIN, HIGH);
  digitalWrite(PWM_PIN, LOW);
  digitalWrite(DIRECTION_PIN, LOW);
  digitalWrite(BRAKE_PIN, HIGH);
  digitalWrite(GND_PIN, LOW);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(PWM_PIN, HIGH);  // turn the LED on (HIGH is the voltage level)
  // digitalWrite(BRAKE_PIN, HIGH);
   delayMicroseconds(high_time);                      // wait for a second
   digitalWrite(PWM_PIN, LOW);   // turn the LED off by making the voltage LOW
   delayMicroseconds(pwm_period - high_time);
}
