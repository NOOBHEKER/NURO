#include "nl_calc_embedded.h"
#include "nl_lcd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define CALC_BUF_SIZE 128
#define PLOT_COLS     46
#define PLOT_ROWS      7

static char s_input_buf[CALC_BUF_SIZE];
static size_t s_buf_idx = 0;
static bool s_active = false;

/* Compact Flash-embedded Periodic Table (H to U) */
typedef struct {
    const char *sym;
    float weight;
} element_t;

static const element_t PROGMEM PERIODIC_TABLE[] = {
    {"H", 1.008f},   {"He", 4.003f},  {"Li", 6.940f},  {"Be", 9.012f},
    {"B", 10.81f},   {"C", 12.011f},  {"N", 14.007f},  {"O", 15.999f},
    {"F", 18.998f},  {"Ne", 20.180f}, {"Na", 22.990f}, {"Mg", 24.305f},
    {"Al", 26.982f}, {"Si", 28.085f}, {"P", 30.974f},  {"S", 32.060f},
    {"Cl", 35.450f}, {"Ar", 39.948f}, {"K", 39.098f},  {"Ca", 40.078f},
    {"Sc", 44.956f}, {"Ti", 47.867f}, {"V", 50.942f},  {"Cr", 51.996f},
    {"Mn", 54.938f}, {"Fe", 55.845f}, {"Co", 58.933f}, {"Ni", 58.693f},
    {"Cu", 63.546f}, {"Zn", 65.380f}, {"Ga", 69.723f}, {"Ge", 72.630f},
    {"As", 74.922f}, {"Se", 78.971f}, {"Br", 79.904f}, {"Kr", 83.798f},
    {"Rb", 85.468f}, {"Sr", 87.620f}, {"Y", 88.906f},  {"Zr", 91.224f},
    {"Nb", 92.906f}, {"Mo", 95.950f}, {"Ag", 107.87f}, {"Cd", 112.41f},
    {"Sn", 118.71f}, {"I", 126.90f},  {"Ba", 137.33f}, {"W", 183.84f},
    {"Pt", 195.08f}, {"Au", 196.97f}, {"Hg", 200.59f}, {"Pb", 207.20f},
    {"U", 238.03f},
    {NULL, 0.0f}
};

static float lookup_element_weight(const char *sym, size_t len) {
    char s[4];
    if (len >= sizeof(s)) len = sizeof(s) - 1;
    memcpy(s, sym, len);
    s[len] = '\0';

    for (size_t i = 0; PERIODIC_TABLE[i].sym != NULL; ++i) {
        if (strcmp(s, PERIODIC_TABLE[i].sym) == 0) {
            return PERIODIC_TABLE[i].weight;
        }
    }
    return 0.0f;
}

/* ASCII 2D Curve Plotter */
static void render_ascii_plot(float *ys, int n, float y_min, float y_max, const char *title) {
    if (fabsf(y_max - y_min) < 1e-6f) {
        y_max += 1.0f;
        y_min -= 1.0f;
    }

    Serial.printf("\r\n+---[ %s ]", title);
    for (int i = 0; i < PLOT_COLS - (int)strlen(title) - 8; ++i) Serial.print("-");
    Serial.print("+\r\n");

    for (int r = 0; r < PLOT_ROWS; ++r) {
        if (r == 0) {
            Serial.printf("| %7.1f |", y_max);
        } else if (r == PLOT_ROWS / 2) {
            Serial.printf("| %7.1f |", (y_max + y_min) * 0.5f);
        } else if (r == PLOT_ROWS - 1) {
            Serial.printf("| %7.1f |", y_min);
        } else {
            Serial.print("|         |");
        }

        for (int c = 0; c < PLOT_COLS; ++c) {
            int y_idx = (c * n) / PLOT_COLS;
            float y_val = ys[y_idx];
            int pt_row = (int)roundf(((y_max - y_val) / (y_max - y_min)) * (PLOT_ROWS - 1));
            if (pt_row == r) {
                Serial.print("*");
            } else if (y_min <= 0.0f && y_max >= 0.0f && r == (int)roundf((y_max / (y_max - y_min)) * (PLOT_ROWS - 1))) {
                Serial.print("-");
            } else {
                Serial.print(" ");
            }
        }
        Serial.print("|\r\n");
    }
    Serial.print("+---------+");
    for (int i = 0; i < PLOT_COLS; ++i) Serial.print("-");
    Serial.print("+\r\n");
}

/* 1. Mathematics: Quadratic Equation Solver */
static void cmd_quadratic(float a, float b, float c) {
    if (fabsf(a) < 1e-7f) {
        Serial.println("  [ERROR]: 'a' coefficient cannot be zero in a quadratic equation.");
        return;
    }

    float disc = b * b - 4.0f * a * c;
    float h_v = -b / (2.0f * a);
    float k_v = c - (b * b) / (4.0f * a);
    char lcd_res[17] = {0};

    Serial.println("\r\n=======================================================");
    Serial.printf("⚡ ON-CHIP QUADRATIC SOLVER: %.2f·x² + (%.2f)·x + (%.2f) = 0\r\n", a, b, c);
    Serial.println("=======================================================");
    Serial.printf("  Discriminant Δ = b² - 4ac : %.4f\r\n", disc);

    if (disc > 0.0f) {
        float sqrt_d = sqrtf(disc);
        float x1 = (-b + sqrt_d) / (2.0f * a);
        float x2 = (-b - sqrt_d) / (2.0f * a);
        Serial.println("  Root Nature               : Two Distinct Real Roots");
        Serial.printf("  Roots                     : x₁ = %.4f,  x₂ = %.4f\r\n", x1, x2);
        snprintf(lcd_res, sizeof(lcd_res), "x1=%.2f x2=%.2f", x1, x2);
    } else if (fabsf(disc) < 1e-6f) {
        float x1 = -b / (2.0f * a);
        Serial.println("  Root Nature               : One Repeated Real Root (Double Root)");
        Serial.printf("  Root                      : x = %.4f\r\n", x1);
        snprintf(lcd_res, sizeof(lcd_res), "Double root=%.2f", x1);
    } else {
        float sqrt_neg_d = sqrtf(-disc);
        float real_p = -b / (2.0f * a);
        float imag_p = sqrt_neg_d / (2.0f * a);
        Serial.println("  Root Nature               : Two Complex Conjugate Roots");
        Serial.printf("  Roots                     : x = %.4f ± %.4f·i\r\n", real_p, fabsf(imag_p));
        snprintf(lcd_res, sizeof(lcd_res), "%.1f +/- %.1fi", real_p, fabsf(imag_p));
    }

    Serial.printf("  Parabola Vertex (h, k)    : (%.3f, %.3f)\r\n", h_v, k_v);
    Serial.printf("  Axis of Symmetry          : x = %.3f\r\n", h_v);

    nl_lcd_show_math("QUADRATIC SOLN", lcd_res);

    /* Generate curve data for ASCII plot */
    float ys[PLOT_COLS];
    float x_start = h_v - 4.0f;
    float x_end   = h_v + 4.0f;
    float y_min = 1e9f, y_max = -1e9f;

    for (int i = 0; i < PLOT_COLS; ++i) {
        float cur_x = x_start + (float)i / (float)(PLOT_COLS - 1) * (x_end - x_start);
        ys[i] = a * cur_x * cur_x + b * cur_x + c;
        if (ys[i] < y_min) y_min = ys[i];
        if (ys[i] > y_max) y_max = ys[i];
    }
    render_ascii_plot(ys, PLOT_COLS, y_min, y_max, "Parabola y = f(x)");
}

/* 2. Physics: Pascal's Law */
static void cmd_pascals_law(float f1, float a1, float a2) {
    if (a1 <= 0.0f || a2 <= 0.0f) {
        Serial.println("  [ERROR]: Areas must be strictly positive.");
        return;
    }
    float p = f1 / a1;
    float ma = a2 / a1;
    float f2 = f1 * ma;
    float stroke_in = 0.10f; // 10 cm input stroke
    float stroke_out = stroke_in / ma;

    Serial.println("\r\n=======================================================");
    Serial.println("⚡ PASCAL'S LAW & HYDRAULIC MECHANICAL ADVANTAGE");
    Serial.println("=======================================================");
    Serial.println("  Principle: Enclosed fluid transmits pressure equally in all directions.");
    Serial.printf("  Input Force F₁            : %.2f N\r\n", f1);
    Serial.printf("  Input Piston Area A₁      : %.4f m²\r\n", a1);
    Serial.printf("  Enclosed Pressure P       : %.2f Pa (%.2f kPa)\r\n", p, p / 1000.0f);
    Serial.printf("  Output Piston Area A₂     : %.4f m²\r\n", a2);
    Serial.printf("  Mechanical Advantage (MA) : %.2fx Force Multiplication\r\n", ma);
    Serial.printf("  Output Lift Force F₂      : %.2f N (Payload: %.1f kg)\r\n", f2, f2 / 9.81f);
    Serial.printf("  Stroke Conservation       : Input 10.0 cm ➔ Lift %.2f cm (Work: %.2f J)\r\n",
                  stroke_out * 100.0f, f1 * stroke_in);

    char l1[17], l2[17];
    snprintf(l1, sizeof(l1), "PASCAL MA: %.1fx", ma);
    snprintf(l2, sizeof(l2), "F2:%.0fN (%.0fkg)", f2, f2 / 9.81f);
    nl_lcd_show_math(l1, l2);

    float ys[PLOT_COLS];
    for (int i = 0; i < PLOT_COLS; ++i) {
        float r = 1.0f + ((float)i / (float)(PLOT_COLS - 1)) * 9.0f; // ratio 1 to 10
        ys[i] = f1 * (r * r); // F2 = F1 * (d2/d1)^2
    }
    render_ascii_plot(ys, PLOT_COLS, ys[0], ys[PLOT_COLS - 1], "Force F2 vs Diameter Ratio");
}

/* 3. Physics: Projectile Ballistics */
static void cmd_projectile(float v0, float theta_deg) {
    float g = 9.80665f;
    float theta_rad = theta_deg * (float)M_PI / 180.0f;
    float vx = v0 * cosf(theta_rad);
    float vy = v0 * sinf(theta_rad);

    float t_flight = (2.0f * vy) / g;
    float h_max    = (vy * vy) / (2.0f * g);
    float range    = vx * t_flight;

    Serial.println("\r\n=======================================================");
    Serial.printf("⚡ PROJECTILE TRAJECTORY: v₀ = %.2f m/s, θ = %.1f°\r\n", v0, theta_deg);
    Serial.println("=======================================================");
    Serial.printf("  Time of Flight            : %.3f s\r\n", t_flight);
    Serial.printf("  Maximum Altitude H_max    : %.2f m\r\n", h_max);
    Serial.printf("  Total Range R             : %.2f m\r\n", range);
    Serial.printf("  Impact Velocity           : %.2f m/s @ -%.1f°\r\n", v0, theta_deg);

    char l1[17], l2[17];
    snprintf(l1, sizeof(l1), "PROJ: v0=%.0f a=%.0f", v0, theta_deg);
    snprintf(l2, sizeof(l2), "R:%.1fm H:%.1fm", range, h_max);
    nl_lcd_show_math(l1, l2);

    float ys[PLOT_COLS];
    for (int i = 0; i < PLOT_COLS; ++i) {
        float cur_x = (float)i / (float)(PLOT_COLS - 1) * range;
        float cur_t = cur_x / (vx > 1e-4f ? vx : 1e-4f);
        ys[i] = vy * cur_t - 0.5f * g * cur_t * cur_t;
        if (ys[i] < 0.0f) ys[i] = 0.0f;
    }
    render_ascii_plot(ys, PLOT_COLS, 0.0f, h_max, "Trajectory y(x) [m]");
}

/* 4. Physics: Series RLC Circuit */
static void cmd_rlc(float R, float L_mH, float C_uF) {
    float L = L_mH * 1e-3f;
    float C = C_uF * 1e-6f;
    if (L <= 0.0f || C <= 0.0f) {
        Serial.println("  [ERROR]: L and C must be positive.");
        return;
    }

    float w0 = 1.0f / sqrtf(L * C);
    float f0 = w0 / (2.0f * (float)M_PI);
    float alpha = R / (2.0f * L);
    float zeta = alpha / w0;
    float Q = (1.0f / R) * sqrtf(L / C);

    Serial.println("\r\n=======================================================");
    Serial.println("⚡ SERIES RLC TRANSIENT & IMPEDANCE ENGINE");
    Serial.println("=======================================================");
    Serial.printf("  Components                : R=%.2f Ω, L=%.2f mH, C=%.2f µF\r\n", R, L_mH, C_uF);
    Serial.printf("  Resonance Frequency f₀    : %.2f Hz (ω₀ = %.2f rad/s)\r\n", f0, w0);
    Serial.printf("  Neper Damping α           : %.2f s⁻¹ (Damping Ratio ζ = %.4f)\r\n", alpha, zeta);
    Serial.printf("  Quality Factor Q          : %.3f\r\n", Q);

    if (zeta < 1.0f) {
        float wd = sqrtf(w0 * w0 - alpha * alpha);
        Serial.printf("  Regime                    : Underdamped Ringing (f_d = %.2f Hz)\r\n", wd / (2.0f * (float)M_PI));
    } else if (fabsf(zeta - 1.0f) < 1e-3f) {
        Serial.println("  Regime                    : Critically Damped");
    } else {
        Serial.println("  Regime                    : Overdamped (Aperiodic)");
    }
}

/* 5. Chemistry: Molar Mass Calculator */
static void cmd_molar_mass(const char *formula) {
    float total_mass = 0.0f;
    size_t len = strlen(formula);
    size_t i = 0;

    typedef struct {
        char sym[4];
        int count;
        float mass;
    } elem_count_t;

    elem_count_t breakdown[16];
    size_t n_elems = 0;

    while (i < len && n_elems < 16) {
        if (!isupper(formula[i])) {
            i++;
            continue;
        }

        char sym[4];
        sym[0] = formula[i++];
        sym[1] = '\0';
        if (i < len && islower(formula[i])) {
            sym[1] = formula[i++];
            sym[2] = '\0';
        }

        int count = 0;
        while (i < len && isdigit(formula[i])) {
            count = count * 10 + (formula[i++] - '0');
        }
        if (count == 0) count = 1;

        float aw = lookup_element_weight(sym, strlen(sym));
        if (aw > 0.0f) {
            size_t cpy_len = sizeof(breakdown[n_elems].sym) - 1;
            strncpy(breakdown[n_elems].sym, sym, cpy_len);
            breakdown[n_elems].sym[cpy_len] = '\0';
            breakdown[n_elems].count = count;
            breakdown[n_elems].mass = aw * (float)count;
            total_mass += breakdown[n_elems].mass;
            n_elems++;
        }
    }

    if (total_mass <= 0.0f) {
        Serial.println("  [ERROR]: Unknown or unparseable chemical formula.");
        return;
    }

    Serial.println("\r\n=======================================================");
    Serial.printf("⚡ FLASH MOLAR MASS & COMPOSITION: %s\r\n", formula);
    Serial.println("=======================================================");
    Serial.printf("  Molar Mass (M)            : %.4f g/mol\r\n", total_mass);
    Serial.println("  Elemental Mass Breakdown  :");
    for (size_t k = 0; k < n_elems; ++k) {
        float pct = (breakdown[k].mass / total_mass) * 100.0f;
        int bars = (int)(pct / 5.0f);
        Serial.printf("    %2s (%2d): %6.2f%% | ", breakdown[k].sym, breakdown[k].count, pct);
        for (int b = 0; b < bars; ++b) Serial.print("=");
        Serial.println();
    }

    char l1[17], l2[17];
    snprintf(l1, sizeof(l1), "MASS:%-11s", formula);
    snprintf(l2, sizeof(l2), "M=%.2f g/mol", total_mass);
    nl_lcd_show_math(l1, l2);
}

/* 6. Chemistry: Solution pH */
static void cmd_ph(float conc, bool is_base) {
    if (conc <= 0.0f) {
        Serial.println("  [ERROR]: Concentration must be strictly positive.");
        return;
    }
    float p_val = -log10f(conc);
    float ph = is_base ? (14.0f - p_val) : p_val;
    float poh = 14.0f - ph;

    Serial.println("\r\n=======================================================");
    Serial.printf("⚡ SOLUTION %s pH / pOH\r\n", is_base ? "STRONG BASE" : "STRONG ACID");
    Serial.println("=======================================================");
    Serial.printf("  Concentration             : %.4e M\r\n", conc);
    Serial.printf("  Calculated pH             : %.3f\r\n", ph);
    Serial.printf("  Calculated pOH            : %.3f\r\n", poh);
}

void nl_calc_init(void) {
    s_buf_idx = 0;
    s_input_buf[0] = '\0';
    s_active = false;
}

bool nl_calc_is_active(void) {
    return s_active;
}

void nl_calc_exec_command(const char *cmd_line) {
    s_active = true;
    while (*cmd_line == ' ') cmd_line++;
    if (*cmd_line == '\0') {
        Serial.print("\r\nesp32-s3> ");
        s_active = false;
        return;
    }

    char cmd[16] = {0};
    const char *args = cmd_line;
    size_t c_idx = 0;
    while (*args && *args != ' ' && c_idx < sizeof(cmd) - 1) {
        cmd[c_idx++] = tolower(*args++);
    }
    cmd[c_idx] = '\0';
    while (*args == ' ') args++;

    if (strcmp(cmd, "help") == 0) {
        Serial.println("\r\n=======================================================");
        Serial.println("⚡ ESP32-S3 ON-CHIP SCIENTIFIC LAB COMMANDS");
        Serial.println("=======================================================");
        Serial.println("  MATH:");
        Serial.println("    quad <a> <b> <c>         : Solve ax²+bx+c=0 with Δ, roots & parabola");
        Serial.println("  PHYSICS:");
        Serial.println("    pascal <F1> <A1> <A2>    : Pascal's Law hydraulic amplification");
        Serial.println("    proj <v0> <angle>        : Ballistics flight time, range & trajectory");
        Serial.println("    rlc <R> <L_mH> <C_uF>    : Series RLC resonant frequency, Q & damping");
        Serial.println("  CHEMISTRY:");
        Serial.println("    mass <formula>           : Flash periodic table molar mass (e.g. C6H12O6)");
        Serial.println("    ph <conc> [base]         : Solution pH/pOH calculation");
        Serial.println("  NEURAL TESTS:");
        Serial.println("    neuro                    : Live on-chip DSP envelope, phase & pulse stats");
        Serial.println("    sim burst                : Self-test 20Hz burst alert (requires 'arm')");
        Serial.println("    sim rest                 : Restore normal baseline noise");
        Serial.println("    status                   : Chip CPU frequency, free heap & PSRAM");
        Serial.println("  SAFETY:");
        Serial.println("    arm                      : Enable GPIO4 stimulation marker (boot = disarmed)");
        Serial.println("    disarm                   : Disable GPIO4 immediately and go safe");
    } else if (strcmp(cmd, "arm") == 0) {
        nl_set_armed(true);
    } else if (strcmp(cmd, "disarm") == 0) {
        nl_set_armed(false);
    } else if (strcmp(cmd, "quad") == 0) {
        float a = 1.0f, b = 0.0f, c = 0.0f;
        if (sscanf(args, "%f %f %f", &a, &b, &c) >= 3) {
            cmd_quadratic(a, b, c);
        } else {
            Serial.println("  Usage: quad <a> <b> <c>  (e.g. quad 1 -5 6)");
        }
    } else if (strcmp(cmd, "pascal") == 0) {
        float f1 = 50.0f, a1 = 0.01f, a2 = 0.5f;
        if (sscanf(args, "%f %f %f", &f1, &a1, &a2) >= 3) {
            cmd_pascals_law(f1, a1, a2);
        } else {
            Serial.println("  Usage: pascal <F1_N> <A1_m2> <A2_m2>  (e.g. pascal 50 0.01 0.5)");
        }
    } else if (strcmp(cmd, "proj") == 0) {
        float v0 = 50.0f, ang = 45.0f;
        if (sscanf(args, "%f %f", &v0, &ang) >= 2) {
            cmd_projectile(v0, ang);
        } else {
            Serial.println("  Usage: proj <v0_m/s> <angle_deg>  (e.g. proj 50 45)");
        }
    } else if (strcmp(cmd, "rlc") == 0) {
        float R = 5.0f, L = 50.0f, C = 100.0f;
        if (sscanf(args, "%f %f %f", &R, &L, &C) >= 3) {
            cmd_rlc(R, L, C);
        } else {
            Serial.println("  Usage: rlc <R_ohm> <L_mH> <C_uF>  (e.g. rlc 5 50 100)");
        }
    } else if (strcmp(cmd, "mass") == 0) {
        if (*args) {
            cmd_molar_mass(args);
        } else {
            Serial.println("  Usage: mass <formula>  (e.g. mass C6H12O6, mass CuSO4)");
        }
    } else if (strcmp(cmd, "ph") == 0) {
        float conc = 0.01f;
        char flag[16] = {0};
        int parsed = sscanf(args, "%f %15s", &conc, flag);
        if (parsed >= 1) {
            bool is_b = (parsed >= 2 && strcasecmp(flag, "base") == 0);
            cmd_ph(conc, is_b);
        } else {
            Serial.println("  Usage: ph <molarity> [base]  (e.g. ph 0.01, ph 0.05 base)");
        }
    } else if (strcmp(cmd, "neuro") == 0) {
        nl_live_telemetry_t telem;
        nl_get_telemetry(&telem);
        float phase_deg = ((float)telem.last_phase_q15 * 180.0f) / 32768.0f;

        Serial.println("\r\n=======================================================");
        Serial.println("⚡ ESP32-S3 CLOSED-LOOP NEUROMODULATION TELEMETRY");
        Serial.println("=======================================================");
        Serial.printf("  Filter Pipeline           : 61-tap Q15 Quadrature FIR @ 1000 Hz\r\n");
        Serial.printf("  Classifier Network        : 64-neuron Int8 Leaky Integrate-and-Fire SNN\r\n");
        Serial.printf("  Loop Controller State     : %s (0x%02X)\r\n",
                      (telem.state == 0) ? "DISARMED" :
                      (telem.state == 1) ? "ARMED" :
                      (telem.state == 2) ? "SUPPRESS" : "REFRACTORY",
                      telem.state);
        Serial.printf("  Stimulation Gate          : %s\r\n", nl_is_armed() ? "ARMED (GPIO4 live)" : "DISARMED (GPIO4 safe)");
        Serial.printf("  Exponential Mean Envelope : %ld\r\n", (long)telem.env_mean_q15);
        Serial.printf("  Instantaneous Envelope    : %d\r\n", (int)telem.last_env_q15);
        Serial.printf("  Compensated Phase Angle   : %+.1f° (%d Q15)\r\n", phase_deg, (int)telem.last_phase_q15);
        Serial.printf("  LIF Hidden Spike Count    : %d / 8 steps\r\n", (int)telem.last_spikes);
        Serial.printf("  Stimulus Budget Consumed  : %u / 300 pulses\r\n", (unsigned int)telem.budget_used);
        Serial.printf("  20ms Artifact Blanking    : %s\r\n", telem.is_blanked ? "ENGAGED" : "IDLE");

        nl_lcd_show_neuro((telem.state == 1) ? "ARMED" : "ACTIVE", telem.last_env_q15, (int16_t)phase_deg);
    } else if (strcmp(cmd, "sim") == 0) {
        if (strcasecmp(args, "burst") == 0) {
            nl_run_sim_burst();
        } else if (strcasecmp(args, "rest") == 0) {
            nl_run_sim_rest();
        } else {
            Serial.println("  Usage: sim <burst|rest>  (e.g. sim burst)");
        }
    } else if (strcmp(cmd, "burst") == 0) {
        nl_run_sim_burst();
    } else if (strcmp(cmd, "rest") == 0) {
        nl_run_sim_rest();
    } else if (strcmp(cmd, "status") == 0) {
        Serial.println("\r\n=======================================================");
        Serial.println("⚡ ESP32-S3 SYSTEM HARDWARE STATUS");
        Serial.println("=======================================================");
        Serial.printf("  CPU Frequency             : %lu MHz\r\n", (unsigned long)getCpuFrequencyMhz());
        Serial.printf("  Free Internal SRAM        : %lu bytes\r\n", (unsigned long)ESP.getFreeHeap());
        Serial.printf("  Free Octal PSRAM          : %lu bytes\r\n", (unsigned long)ESP.getFreePsram());
        Serial.printf("  Flash Chip Size           : %lu MB\r\n", (unsigned long)(ESP.getFlashChipSize() / (1024 * 1024)));
        Serial.printf("  Stimulation Gate          : %s\r\n", nl_is_armed() ? "ARMED (GPIO4 live)" : "DISARMED (GPIO4 safe)");
        Serial.printf("  System Uptime             : %.2f seconds\r\n", millis() / 1000.0f);
        Serial.printf("  I2C 16x2 LCD              : %s\r\n", nl_lcd_is_present() ? "Detected (0x27/0x3F)" : "None (Silent)");
    } else {
        Serial.printf("  [?] Unknown command: '%s'. Type 'help' for full command list.\r\n", cmd);
    }

    Serial.print("\r\nesp32-s3> ");
    s_active = false;
}

void nl_calc_process_char(char c) {
    if (c == '\r' || c == '\n') {
        if (s_buf_idx > 0) {
            s_input_buf[s_buf_idx] = '\0';
            Serial.println();
            nl_calc_exec_command(s_input_buf);
            s_buf_idx = 0;
            s_input_buf[0] = '\0';
        } else {
            Serial.print("\r\nesp32-s3> ");
        }
    } else if (c == '\b' || c == 0x7F) { // Backspace
        if (s_buf_idx > 0) {
            s_buf_idx--;
            Serial.print("\b \b");
        }
    } else if (s_buf_idx < CALC_BUF_SIZE - 1 && c >= 32 && c <= 126) {
        s_input_buf[s_buf_idx++] = c;
        Serial.print(c); // Echo character
    }
}
