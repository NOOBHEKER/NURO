#ifndef NL_LCD_H
#define NL_LCD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize I2C and probe for 16x2 LCD backpack at 0x27 or 0x3F on SDA=GPIO 8, SCL=GPIO 9 */
void nl_lcd_init(void);

/* Check if physical LCD was detected on I2C bus */
bool nl_lcd_is_present(void);

/* Display two 16-character lines on LCD */
void nl_lcd_print(const char *line1, const char *line2);

/* Format helper for Scientific Lab results */
void nl_lcd_show_math(const char *title, const char *result);
void nl_lcd_show_neuro(const char *status, int16_t env, int16_t phase_deg);

#ifdef __cplusplus
}
#endif

#endif /* NL_LCD_H */
