int PWM_PIN = 8;
int DIRECTION_PIN = 9;
int BREAK_PIN = 10;
int FAULT_PIN = 11;
int ENABLE_PIN = 12;
int ADC_PIN = A0;

int pwm_period = 15;
int high_time = pwm_period * 0.2;

void setup() {
  // Set port directions
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIRECTION_PIN, OUTPUT);
  pinMode(BREAK_PIN, OUTPUT);
  pinMode(FAULT_PIN, INPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  // Initial pin values
  digitalWrite(ENABLE_PIN, HIGH);
  digitalWrite(PWM_PIN, LOW);
  digitalWrite(DIRECTION_PIN, LOW);
  digitalWrite(BREAK_PIN, HIGH);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(PWM_PIN, HIGH);  // turn the LED on (HIGH is the voltage level)
  delayMicroseconds(high_time);                      // wait for a second
  digitalWrite(PWM_PIN, LOW);   // turn the LED off by making the voltage LOW
  delayMicroseconds(pwm_period - high_time);
}
