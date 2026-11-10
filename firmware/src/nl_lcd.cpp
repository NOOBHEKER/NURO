#include "nl_lcd.h"
#include <Arduino.h>
#include <Wire.h>
#include <string.h>
#include <stdio.h>

#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9

/* Standard PCF8574 pin mapping to HD44780:
 * P0: RS (0=Command, 1=Data)
 * P1: RW (0=Write)
 * P2: EN (Enable pulse)
 * P3: Backlight (1=On)
 * P4: D4
 * P5: D5
 * P6: D6
 * P7: D7
 */
#define LCD_RS        0x01
#define LCD_EN        0x04
#define LCD_BACKLIGHT 0x08

static uint8_t s_i2c_addr = 0;
static bool s_lcd_present = false;

static void lcd_write_nibble(uint8_t nibble, uint8_t rs) {
    if (!s_lcd_present) return;
    uint8_t data = (nibble & 0xF0) | LCD_BACKLIGHT | (rs ? LCD_RS : 0);
    
    Wire.beginTransmission(s_i2c_addr);
    Wire.write(data | LCD_EN);
    Wire.endTransmission();
    delayMicroseconds(2);

    Wire.beginTransmission(s_i2c_addr);
    Wire.write(data & ~LCD_EN);
    Wire.endTransmission();
    delayMicroseconds(50);
}

static void lcd_send_byte(uint8_t val, uint8_t rs) {
    if (!s_lcd_present) return;
    lcd_write_nibble(val & 0xF0, rs);
    lcd_write_nibble((val << 4) & 0xF0, rs);
}

static void lcd_cmd(uint8_t cmd) {
    lcd_send_byte(cmd, 0);
    if (cmd == 0x01 || cmd == 0x02) {
        delay(2);
    }
}

static void lcd_data(uint8_t data) {
    lcd_send_byte(data, 1);
}

void nl_lcd_init(void) {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 100000);
    delay(50);

    /* Probe 0x27 first, then 0x3F */
    Wire.beginTransmission(0x27);
    if (Wire.endTransmission() == 0) {
        s_i2c_addr = 0x27;
        s_lcd_present = true;
    } else {
        Wire.beginTransmission(0x3F);
        if (Wire.endTransmission() == 0) {
            s_i2c_addr = 0x3F;
            s_lcd_present = true;
        }
    }

    if (!s_lcd_present) {
        return; /* No physical LCD attached, silent bypass */
    }

    /* HD44780 4-bit initialization sequence */
    delay(50);
    lcd_write_nibble(0x30, 0);
    delay(5);
    lcd_write_nibble(0x30, 0);
    delayMicroseconds(150);
    lcd_write_nibble(0x30, 0);
    delayMicroseconds(150);
    lcd_write_nibble(0x20, 0); /* 4-bit mode */
    delayMicroseconds(150);

    lcd_cmd(0x28); /* 2 lines, 5x8 font */
    lcd_cmd(0x0C); /* Display ON, cursor OFF, blink OFF */
    lcd_cmd(0x06); /* Entry mode: auto-increment cursor */
    lcd_cmd(0x01); /* Clear display */
    delay(2);

    nl_lcd_print("NEURAL LOOP S3", "SYS ARMED 1000Hz");
}

bool nl_lcd_is_present(void) {
    return s_lcd_present;
}

void nl_lcd_print(const char *line1, const char *line2) {
    if (!s_lcd_present) return;

    /* Line 1: DDRAM 0x80 */
    lcd_cmd(0x80);
    size_t len1 = line1 ? strlen(line1) : 0;
    for (size_t i = 0; i < 16; ++i) {
        lcd_data((i < len1) ? (uint8_t)line1[i] : ' ');
    }

    /* Line 2: DDRAM 0xC0 */
    lcd_cmd(0xC0);
    size_t len2 = line2 ? strlen(line2) : 0;
    for (size_t i = 0; i < 16; ++i) {
        lcd_data((i < len2) ? (uint8_t)line2[i] : ' ');
    }
}

void nl_lcd_show_math(const char *title, const char *result) {
    if (!s_lcd_present) return;
    nl_lcd_print(title, result);
}

void nl_lcd_show_neuro(const char *status, int16_t env, int16_t phase_deg) {
    if (!s_lcd_present) return;
    char l1[17], l2[17];
    snprintf(l1, sizeof(l1), "N:%-8s E:%-3d", status, (int)env);
    snprintf(l2, sizeof(l2), "PHASE: %+4d deg", (int)phase_deg);
    nl_lcd_print(l1, l2);
}
