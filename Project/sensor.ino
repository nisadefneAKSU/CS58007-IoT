#include "DHT.h"

#define DHTPIN 2
#define DHTTYPE DHT22
#define PIR_PIN 4
#define MQ_PIN A3
#define LDR_PIN A2

DHT dht(DHTPIN, DHTTYPE);

unsigned long previousMillis = 0;
const long interval = 2000; // 2 saniyede bir veri gönder
bool motionDetectedThisCycle = false; // Bu 2 saniye içinde hareket oldu mu?

void setup() {
  Serial.begin(9600);
  dht.begin();
  pinMode(PIR_PIN, INPUT);
  pinMode(LDR_PIN, INPUT_PULLUP);

}

void loop() {
  // --- PIR KONTROLÜ (Sürekli çalışır) ---
  // Eğer o 2 saniyelik aralıkta bir kez bile hareket olursa, 'true' olarak kilitlenir
  if (digitalRead(PIR_PIN) == HIGH) {
    motionDetectedThisCycle = true;
  }

  // --- ZAMANLAYICI (2 saniye doldu mu kontrolü) ---
  unsigned long currentMillis = millis();
  
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // Verileri oku
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    int gasValue = analogRead(MQ_PIN);
    int lightValue = analogRead(LDR_PIN);

    if (!isnan(h) && !isnan(t)) {
      // Veriyi yazdır
      Serial.print(t);          Serial.print(",");
      Serial.print(h);          Serial.print(",");
      Serial.print(lightValue); Serial.print(",");
      Serial.print(gasValue);   Serial.print(",");
      
      // Bu periyotta hareket olduysa 1, olmadıysa 0 basar
      Serial.println(motionDetectedThisCycle ? 1 : 0);
    }

    // YENİ PERİYOT İÇİN HAREKETİ SIFIRLA
    motionDetectedThisCycle = false;
  }
}