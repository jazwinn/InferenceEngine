#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace ui {

constexpr const char* kReset = "\x1b[0m";
constexpr const char* kBold = "\x1b[1m";
constexpr const char* kDim = "\x1b[2m";
constexpr const char* kAccent = "\x1b[38;5;75m";
constexpr const char* kClearLine = "\r\x1b[2K";

// UTF-8 glyphs spelled as bytes so they survive any source/compiler charset.
constexpr const char* kIcon = "\xE2\x97\x86";    // ◆
constexpr const char* kDot = "\xE2\x97\x8F";     // ●
constexpr const char* kHLine = "\xE2\x94\x80";   // ─
constexpr const char* kVLine = "\xE2\x94\x82";   // │
constexpr const char* kTopL = "\xE2\x95\xAD";    // ╭
constexpr const char* kTopR = "\xE2\x95\xAE";    // ╮
constexpr const char* kBotL = "\xE2\x95\xB0";    // ╰
constexpr const char* kBotR = "\xE2\x95\xAF";    // ╯

constexpr size_t kWidth = 60;

inline void EnableTerminal() {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode = 0;
  if (GetConsoleMode(out, &mode)) SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

inline std::string Repeat(const char* glyph, size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i) s += glyph;
  return s;
}

inline void BoxLine(const std::string& text, size_t visible, const char* style = "") {
  std::cout << kAccent << kVLine << kReset << " " << style << text << kReset
            << std::string(kWidth - 3 - std::min(visible, kWidth - 3), ' ') << kAccent << kVLine << kReset << "\n";
}

inline void Banner(const std::string& modelName) {
  std::cout << "\n" << kAccent << kTopL << Repeat(kHLine, kWidth - 2) << kTopR << kReset << "\n";
  BoxLine(std::string(kAccent) + kIcon + kReset + kBold + " Welcome to InferenceEngine chat!", 34);
  BoxLine("", 0);
  BoxLine("model: " + modelName, 7 + modelName.size(), kDim);
  BoxLine("/clear to reset the conversation, /exit to quit", 47, kDim);
  std::cout << kAccent << kBotL << Repeat(kHLine, kWidth - 2) << kBotR << kReset << "\n\n";
}

inline void Status(const std::string& text) { std::cout << kClearLine << kDim << text << kReset << std::flush; }

inline void Info(const std::string& text) { std::cout << kDim << "  " << text << kReset << "\n\n"; }

inline bool ReadPrompt(std::string& input) {
  std::cout << kDim << Repeat(kHLine, kWidth) << kReset << "\n" << kBold << "> " << kReset << std::flush;
  if (!std::getline(std::cin, input)) return false;
  std::cout << "\n";
  return true;
}

}  // namespace ui
