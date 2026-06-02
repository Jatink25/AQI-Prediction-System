#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

// ============ OLED Setup (SPI) =============
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_DC 9
#define OLED_RESET 8
#define OLED_CS 10

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &SPI, OLED_DC, OLED_RESET, OLED_CS);

// =========== DHT SETUP ===========
#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// =========== PIN DEFINITION ==========
#define MQ135_PIN A0
#define GREEN  3
#define YELLOW 4
#define RED    5
#define BUZZER 6

// ================= SIZES =============
#define SAMPLE_SIZE  10
#define HISTORY_SIZE 30

// ================= MQ135 CORRECTION CONSTANTS =============
// Reference conditions the sensor was characterized at (MQ135 datasheet)
#define TEMP_REF   20.0f   
#define HUM_REF    65.0f  
// Sensitivity coefficients 

#define TEMP_COEFF 0.02f   // 2% per °C deviation
#define HUM_COEFF  0.008f  // 0.8% per % humidity deviation

// ================= FIRE DETECTION =============

#define FIRE_TEMP_THRESHOLD  50.0f   
#define FIRE_BEEP_ON         100     
#define FIRE_BEEP_OFF        100     

// ================= GLOBAL VARIABLES =============
uint16_t predictionHistory[HISTORY_SIZE];
uint8_t  historyIndex  = 0;
int8_t   realAccuracy  = 0;
int16_t  predictedAQI  = 0;
int16_t  diff          = 0;
const char* warning    = "Stable";

int16_t  readings[SAMPLE_SIZE];
uint8_t  readIndex     = 0;
int32_t  total         = 0;
int16_t  averageGas    = 0;


float    correctedGas  = 0;

int16_t  AQI           = 0;
float    temperature   = 0;
float    humidity      = 0;

int16_t  AQI_prev1     = 0;
int16_t  AQI_prev2     = 0;

int16_t delta_AQI   = 0;
int16_t delta2_AQI  = 0;

bool     historyInitialized = false;

// Fire detection state
bool     fireAlert         = false;
bool     fireBuzzerState   = false;       
uint32_t lastFireBeepTime  = 0;           

// ===================== SETUP =====================
void setup() {
  Serial.begin(9600);

  Serial.println(F("Time,Temp,Humidity,RawGas,CorrectedGas,AQI,PredictedAQI"));

  dht.begin();

  for (uint8_t i = 0; i < SAMPLE_SIZE; i++) {
    readings[i] = 0;
  }

  pinMode(GREEN,  OUTPUT);
  pinMode(YELLOW, OUTPUT);
  pinMode(RED,    OUTPUT);
  pinMode(BUZZER, OUTPUT);

  if (!display.begin(SSD1306_SWITCHCAPVCC)) {
    Serial.println(F("OLED not found"));
    while (true);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println(F("AQI Monitor"));
  display.println(F("Starting..."));
  display.display();

  delay(2000);
}

// ===================== LOOP =====================
void loop() {

  // -------- Read MQ135 (rolling average) --------
  total -= readings[readIndex];
  readings[readIndex] = analogRead(MQ135_PIN);
  total += readings[readIndex];
  readIndex++;
  if (readIndex >= SAMPLE_SIZE) readIndex = 0;

  averageGas = total / SAMPLE_SIZE;

  // -------- Read DHT first — needed for correction --------

  temperature = dht.readTemperature();
  humidity    = dht.readHumidity();
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println(F("DHT Sensor Error!"));
    return;
  }

  // -------- MQ135 Temperature & Humidity Correction --------

  // correctionFactor > 1 when hot/humid  → dividing brings reading DOWN
  // correctionFactor < 1 when cold/dry   → dividing brings reading UP
  float correctionFactor = 1.0f
                         + TEMP_COEFF * (temperature - TEMP_REF)
                         + HUM_COEFF  * (humidity    - HUM_REF);

  // Safety clamp — prevents divide-by-zero or negative correction
  if (correctionFactor < 0.1f) correctionFactor = 0.1f;

  correctedGas = (float)averageGas / correctionFactor;

  // -------- Map corrected reading to AQI --------

  AQI = (int16_t)map((long)correctedGas, 200, 800, 0, 500);
  AQI = constrain(AQI, 0, 500);

  // -------- Initialize lag history on first run --------
  if (!historyInitialized) {
    AQI_prev1 = AQI;
    AQI_prev2 = AQI;
    historyInitialized = true;
  }

  // -------- Trend Features --------
  delta_AQI  = AQI - AQI_prev1;
  delta2_AQI = AQI_prev1 - AQI_prev2;

  // -------- Predict future AQI --------
  predictedAQI = (int16_t)( + 0.98663f * AQI 
                  + 0.40779f * AQI_prev1 
                  - 0.29959f * AQI_prev2 
                  + 0.57885f * delta_AQI 
                  - 0.12853f * delta2_AQI 
                  + 0.34508f * temperature 
                  + 0.11230f * humidity 
                  - 17.91148f ); 
  predictedAQI = constrain(predictedAQI, 0, 500);

  // -------- Accuracy: compare past prediction vs current AQI --------
  int16_t pastPrediction = (int16_t)predictionHistory[historyIndex];
  predictionHistory[historyIndex] = (uint16_t)predictedAQI;

  int16_t err = abs(pastPrediction - AQI);
  realAccuracy = (int8_t)constrain(100 - (err / 5), 0, 100);

  historyIndex++;
  if (historyIndex >= HISTORY_SIZE) historyIndex = 0;

  // -------- Warning: trend detection --------
  diff = predictedAQI - AQI;
  if      (diff >  15) warning = "Rising";
  else if (diff < -15) warning = "Falling";
  else                 warning = "Stable";

  // -------- Fire Detection --------
 
  fireAlert = (temperature >= FIRE_TEMP_THRESHOLD);

  // -------- LED + Buzzer --------
  digitalWrite(GREEN,  LOW);
  digitalWrite(YELLOW, LOW);
  digitalWrite(RED,    LOW);

  const char* airStatus;

  if (fireAlert) {
    digitalWrite(RED, HIGH);
    airStatus = "FIRE!";
  }
  else if (AQI <= 100) {
    digitalWrite(GREEN, HIGH);
    digitalWrite(BUZZER, LOW);
    airStatus = "Good";
  }
  else if (AQI <= 300) {
    // Moderate: single short beep once per loop cycle (every 2s)
    digitalWrite(YELLOW, HIGH);
    digitalWrite(BUZZER, HIGH);
    delay(200);
    digitalWrite(BUZZER, LOW);
    airStatus = "Moderate";
  }
  else {
    // Poor: buzzer stays on continuously
    digitalWrite(RED,    HIGH);
    digitalWrite(BUZZER, HIGH);
    airStatus = "Poor";
  }

  // -------- Fire Buzzer: rapid beep via millis() --------

  if (fireAlert) {
    uint32_t now = millis();
    uint16_t interval = fireBuzzerState ? FIRE_BEEP_ON : FIRE_BEEP_OFF;
    if (now - lastFireBeepTime >= interval) {
      fireBuzzerState   = !fireBuzzerState;
      lastFireBeepTime  = now;
      digitalWrite(BUZZER, fireBuzzerState ? HIGH : LOW);
    }
  } else {
    // Reset fire buzzer state when no longer in fire alert
    fireBuzzerState  = false;
    lastFireBeepTime = 0;
  }

  // -------- OLED Display --------
  display.clearDisplay();
  display.setCursor(0, 0);

  if (fireAlert) {
    display.setTextSize(2);
    display.setCursor(4, 4);
    display.println(F("!! FIRE !!"));   
    display.setCursor(16, 22);
    display.println(F("ALERT !!"));     

    // Temp and evacuate below in normal size
    display.setTextSize(1);
    display.setCursor(0, 42);
    display.print(F("Temp: "));
    display.print(temperature, 1);
    display.println(F(" C"));
    display.setCursor(0, 54);
    display.println(F("!! Evacuate area!"));
  } else {
    // Normal display
    display.print(F("Temp: "));
    display.print(temperature, 1);
    display.println(F(" C"));

    display.print(F("Hum: "));
    display.print(humidity, 1);
    display.println(F(" %"));

    display.print(F("AQI: "));
    display.println(AQI);

    display.print(F("Pred: "));
    display.println(predictedAQI);

    display.print(F("Trend: "));
    display.println(warning);

    display.print(F("Status: "));
    display.println(airStatus);

    display.print(F("Acc: "));
    display.print(realAccuracy);
    display.println(F("%"));
  }

  display.display();

  // -------- Serial CSV Output --------
  Serial.print(millis() / 1000);
  Serial.print(F(","));
  Serial.print(temperature, 1);
  Serial.print(F(","));
  Serial.print(humidity, 1);
  Serial.print(F(","));
  Serial.print(averageGas);         
  Serial.print(F(","));
  Serial.print(correctedGas, 1);    
  Serial.print(F(","));
  Serial.print(AQI);
  Serial.print(F(","));
  Serial.println(predictedAQI);

  // -------- Update lag history --------
  AQI_prev2 = AQI_prev1;
  AQI_prev1 = AQI;

  
  if (!fireAlert) {
    delay(2000);
  }
}
