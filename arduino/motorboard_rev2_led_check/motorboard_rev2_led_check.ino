int BRAKE_PIN = 0;
int PWM_PIN = 1;
int ENABLE_PIN = 2;
int DIRECTION_PIN = 3;
int FAULT_PIN = 4;
int ADC_PIN = 5;
int GND_PIN = 6;
int PWR_PIN = 7;

void setup() {
  // Set port directions
  pinMode(BRAKE_PIN, OUTPUT);
  pinMode(PWM_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(DIRECTION_PIN, OUTPUT);
  pinMode(FAULT_PIN, INPUT);
  pinMode(ADC_PIN, INPUT);
  pinMode(GND_PIN, OUTPUT);
  pinMode(PWR_PIN, OUTPUT);  

  // Initial pin values
  digitalWrite(BRAKE_PIN, HIGH);
  digitalWrite(PWM_PIN, LOW);
  digitalWrite(ENABLE_PIN, LOW);
  digitalWrite(DIRECTION_PIN, LOW);
  digitalWrite(GND_PIN, LOW);
  digitalWrite(PWR_PIN, HIGH); 
  
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(1000);
  digitalWrite(ENABLE_PIN, HIGH);

  // delay(1000);
  // digitalWrite(BRAKE_PIN, HIGH);

  delay(1000);
  digitalWrite(PWM_PIN, HIGH);

  // delay(1000);
  // digitalWrite(DIRECTION_PIN, HIGH);

  delay(1000);
  // digitalWrite(BRAKE_PIN, LOW);
  digitalWrite(PWM_PIN, LOW);
  digitalWrite(ENABLE_PIN, LOW);
  digitalWrite(DIRECTION_PIN, LOW);
}
