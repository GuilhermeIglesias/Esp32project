#include "esp_camera.h"
#include <WiFi.h>

/* ===== WiFi ===== */
const char* ssid = "COMP 2179";
const char* password = "N35588@s";

/* ===== PC (servidor Python) ===== */
const char* serverIP = "194.210.89.70";   // MUDE para o IP REAL do PC
const int serverPort = 5000;

WiFiClient client;

/* ===== Camera pinout: AI-Thinker ===== */
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void setup() {
  Serial.begin(115200);
  Serial.println("\nIniciando ESP32-CAM...");

  /* ===== WiFi ===== */
  WiFi.begin(ssid, password);
  Serial.print("Conectando ao WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi conectado!");
  Serial.print("IP da ESP32: ");
  Serial.println(WiFi.localIP());

  /* ===== Camera config ===== */
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset   = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size   = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("❌ Falha ao inicializar a câmera");
    while (true);
  }

  Serial.println("✅ Câmera pronta");
}

void loop() {

  if (!client.connect(serverIP, serverPort)) {
    Serial.println("❌ Não conectou ao servidor");
    delay(1000);
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Falha ao capturar imagem");
    client.stop();
    return;
  }

  uint32_t size = fb->len;

  client.write((uint8_t*)&size, sizeof(size));
  client.write(fb->buf, fb->len);

  esp_camera_fb_return(fb);
  client.stop();

  Serial.println("📷 Imagem enviada");

  delay(100);
}
