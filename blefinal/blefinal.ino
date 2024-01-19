#define USE_NIMBLE
#include <BleKeyboard.h>
BleKeyboard bleKeyboard;
//BleKeyboard bleKeyboard("Media_Flow", "KK", 100);

#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  Serial.begin(115200);
  lcd.begin();
  lcd.backlight();
  lcd.print("Media Flow");
  Serial.println("Starting BLE work!");
  bleKeyboard.begin();
}

void loop() {
  if(bleKeyboard.isConnected()) {
    char input = Serial.read();
    switch (input) {
      case '1':
        lcd.clear();
        lcd.print("UP");
        bleKeyboard.press(KEY_UP_ARROW);
        bleKeyboard.releaseAll();
        break;
      case '2':
        lcd.clear();
        lcd.print("DOWN");
        bleKeyboard.press(KEY_DOWN_ARROW);
        bleKeyboard.releaseAll();
        break;
      case '3':
        lcd.clear();
        lcd.print("LEFT");
        bleKeyboard.press(KEY_LEFT_ARROW);
        bleKeyboard.releaseAll();
        break;
       case '4':
        lcd.clear();
        lcd.print("RIGHT");
        bleKeyboard.press(KEY_RIGHT_ARROW);
        bleKeyboard.releaseAll();
        break;
       case '5':
        lcd.clear();
        lcd.print("SELECT");
        bleKeyboard.press(KEY_RETURN);
        bleKeyboard.releaseAll();
        break;
       case '6':
        lcd.clear();
        lcd.print("PAUSE/PLAY");
        bleKeyboard.press(' ');
        bleKeyboard.releaseAll();
        break;
       case '7':
        lcd.clear();
        lcd.print("BACK");
        bleKeyboard.press(KEY_ESC);
        bleKeyboard.releaseAll();
        break;
       case '8':
        lcd.clear();
        lcd.print("HOME");
        bleKeyboard.press(KEY_HOME);
        bleKeyboard.releaseAll();
        break;
       case '9':
        lcd.clear();
        lcd.print("VOL_UP");
        bleKeyboard.write(KEY_MEDIA_VOLUME_UP);
        bleKeyboard.releaseAll();
        break;
       case '0':
        lcd.clear();
        lcd.print("VOL_DOWN");
        bleKeyboard.write(KEY_MEDIA_VOLUME_DOWN);
        bleKeyboard.releaseAll();
        break;
      default:
        // Ignore other input
        break;
    }
  }
}
